"""Companion Agent Runner —— 把我们的 ③ agent 接进 AstrBot 的流式 runner 体系。

放到 AstrBot: astrbot/core/agent/runners/companion/companion_agent_runner.py

它连 ③ 的 SSE 端点 `POST {AGENT_BASE}/sns/stream`,把 ③ 的事件流
(token / round_end / tool_call / file / message_end)映射成 AstrBot 的 AgentResponse:
  - token            → streaming_delta(逐字,给平台流式文本)
  - 中间轮 round_end  → 【只在非流式时】直接 event.send(流式下 token 已发过,再发是重复)
  - file             → 直接 event.send(图片/文件组件,AstrBot 用 NapCat 发)
  - notice           → 直接 event.send(纯文本,如 NSFW 图片的原链接)
  - 最终轮 content    → final_llm_resp(收尾)

注意:中间产物【不能】用 AgentResponse(type="llm_result") ——
上游会在我们这条链路上把它全部丢弃,详见 _send_now 的注释。

关键点:说话人 qq / 会话id / 群号 全部从 run_context.context.event 拿,
不需要 ③ 反向穿参;③ 那边 onebot=None,工具只产出 file 事件,由 AstrBot 发送。
"""

from __future__ import annotations

import json
import os
import typing as T

import aiohttp

import astrbot.core.message.components as Comp
from astrbot import logger
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from ...hooks import BaseAgentRunHooks
from ...response import AgentResponse
from ...run_context import ContextWrapper, TContext
from ..base import AgentState, BaseAgentRunner


class CompanionAgentRunner(BaseAgentRunner[TContext]):
    """连接 companion-agent(③)的 SSE 流式 runner。"""

    async def reset(
        self,
        request: ProviderRequest,
        run_context: ContextWrapper[TContext],
        agent_hooks: BaseAgentRunHooks[TContext],
        provider_config: dict,
        **kwargs: T.Any,
    ) -> None:
        self.req = request
        self.run_context = run_context
        self.agent_hooks = agent_hooks
        self.final_llm_resp: LLMResponse | None = None
        self._state = AgentState.IDLE
        # 流式开启时,每一轮(含中间轮)的正文已经由 token → streaming_delta 逐字发出去了,
        # 中间轮的 round_end 再发一次就是【重复发言】。关掉流式时 token 不落地,
        # 中间发言才需要我们自己补发。
        self.streaming: bool = bool(kwargs.get("streaming", True))
        # ③ 的地址 + api key。优先读 AstrBot 的 provider 配置,没有就退回环境变量,
        # 再没有就用默认地址。这样最省事:直接 export COMPANION_AGENT_BASE /
        # COMPANION_API_KEY 就能用,不用非在 AstrBot 配置里塞自定义键。
        self.agent_base: str = (
            provider_config.get("companion_agent_base")
            or os.environ.get("COMPANION_AGENT_BASE")
            or "http://127.0.0.1:9560"
        ).rstrip("/")
        self.api_key: str = (
            provider_config.get("companion_api_key")
            or os.environ.get("COMPANION_API_KEY", "")
        )

    async def _send_now(self, chain: MessageChain) -> None:
        """立刻把一条消息(中间发言 / 图片 / 文件)发到群或好友。

        为什么不走 `yield AgentResponse(type="llm_result")`:
        上游的 `run_third_party_agent` 只在 `stream_to_general` 为真时才 yield
        llm_result,而流式路径是写死 `stream_to_general=False` 调用的,
        非流式路径下它也只有在「平台不支持流式 + 策略为 turn_off」时才为真。
        也就是说 llm_result 在我们这条链路上【必然被丢弃】。

        看 dify/deerflow/coze 就明白了:它们只在【最终结果】那一下发 llm_result,
        并同时设置 final_llm_resp —— 上游把 llm_result 当"最终结果"用,
        而不是"一条中间消息"。中间产物没有管线通道,只能自己发。
        """
        event = self.run_context.context.event
        try:
            await event.send(chain)
        except Exception as e:  # noqa: BLE001 — 单条发送失败不该中断整轮
            logger.error(f"CompanionAgentRunner failed to send chain: {e}")

    def _build_payload(self) -> dict:
        # run_context.context.event = AstrMessageEvent,元数据都在这
        event = self.run_context.context.event
        is_private = event.is_private_chat()
        sender_id = event.get_sender_id()
        # 会话key:私聊=本人qq,群聊=群号(与 ③ 的 conversation_id 约定对齐)
        session_id = sender_id if is_private else event.get_group_id()
        return {
            "message_type": "private" if is_private else "group",
            "session_id": str(session_id),
            "sender": {"id": str(sender_id), "nickname": event.get_sender_name()},
            "text": self.req.prompt or "",
            "image_urls": list(self.req.image_urls or []),
            "api_key": self.api_key,
        }

    async def step(self) -> T.AsyncGenerator[AgentResponse, None]:
        # 本 runner 一次跑完,step 直接委托给 step_until_done。
        async for resp in self.step_until_done(max_step=1):
            yield resp

    async def step_until_done(
        self, max_step: int = 30
    ) -> T.AsyncGenerator[AgentResponse, None]:
        self._transition_state(AgentState.RUNNING)
        payload = self._build_payload()
        final_content = ""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.agent_base}/sns/stream", json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    resp.raise_for_status()
                    async for raw in _iter_sse(resp):
                        if raw == "[DONE]":
                            break
                        ev = json.loads(raw)
                        etype = ev.get("event")

                        if etype == "token":
                            text = ev.get("text") or ""
                            if text:
                                yield AgentResponse(
                                    type="streaming_delta",
                                    data={"chain": MessageChain().message(text)},
                                )
                        elif etype == "round_end":
                            content = (ev.get("content") or "").strip()
                            if ev.get("final"):
                                final_content = content
                            elif content and not self.streaming:
                                # 中间发言。流式开着时这段正文已经逐字发过了(token →
                                # streaming_delta),这里再发就是重复;只有关掉流式、
                                # token 不落地时才需要补发。
                                await self._send_now(
                                    MessageChain().message(content))
                        elif etype == "file":
                            chain = _file_to_chain(ev.get("file") or {})
                            if chain is not None:
                                await self._send_now(chain)
                        elif etype == "notice":
                            # 工具要求立刻发的纯文本(如 NSFW 图片的原链接):直接发,
                            # ③ 那边不落库、也不进模型上下文。
                            text = (ev.get("text") or "").strip()
                            if text:
                                await self._send_now(MessageChain().message(text))
                        elif etype == "message_end":
                            final_content = ev.get("content") or final_content
                        elif etype == "error":
                            msg = ev.get("message") or "agent error"
                            self.final_llm_resp = LLMResponse(
                                role="err",
                                result_chain=MessageChain().message(f"（出错了：{msg}）"))
                            self._transition_state(AgentState.ERROR)
                            yield AgentResponse(
                                type="err",
                                data={"chain": MessageChain().message(f"（出错了：{msg}）")})
                            return
        except Exception as e:  # noqa: BLE001
            logger.error(f"CompanionAgentRunner error: {e}")
            self.final_llm_resp = LLMResponse(
                role="err", result_chain=MessageChain().message(f"（连接 agent 失败：{e}）"))
            self._transition_state(AgentState.ERROR)
            yield AgentResponse(
                type="err",
                data={"chain": MessageChain().message(f"（连接 agent 失败：{e}）")})
            return

        self.final_llm_resp = LLMResponse(
            role="assistant",
            result_chain=MessageChain().message(final_content) if final_content
            else MessageChain(chain=[]))
        self._transition_state(AgentState.DONE)

    def done(self) -> bool:
        return self._state in (AgentState.DONE, AgentState.ERROR)

    def get_final_llm_resp(self) -> LLMResponse | None:
        return self.final_llm_resp


def _file_to_chain(file: dict) -> MessageChain | None:
    """③ 的 FileRef → AstrBot 消息链(图片走 Image,其它走 File)。"""
    url = file.get("url")
    if not url:
        return None
    name = file.get("name") or "file"
    if file.get("kind") == "image":
        return MessageChain(chain=[Comp.Image.fromURL(url)])
    return MessageChain(chain=[Comp.File(name=name, url=url)])


async def _iter_sse(resp: "aiohttp.ClientResponse") -> T.AsyncGenerator[str, None]:
    """把 SSE 响应按 `data:` 行拆出 payload 文本。"""
    async for line_bytes in resp.content:
        line = line_bytes.decode("utf-8", errors="replace").strip()
        if line.startswith("data:"):
            yield line[5:].strip()
