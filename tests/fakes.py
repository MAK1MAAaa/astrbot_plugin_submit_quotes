"""Small AstrBot/OneBot contracts for offline tests."""

from __future__ import annotations

import base64
import inspect
import logging
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import unquote, urlparse


class Plain:
    def __init__(self, text: str):
        self.text = text

    async def to_dict(self) -> dict:
        return {"type": "text", "data": {"text": self.text}}


class Reply:
    def __init__(self, id: str | int):
        self.id = id


class At:
    def __init__(self, qq: str | int, name: str = ""):
        self.qq, self.name = qq, name

    async def to_dict(self) -> dict:
        return {"type": "at", "data": {"qq": str(self.qq)}}


class Face:
    def __init__(self, id: int):
        self.id = id

    async def to_dict(self) -> dict:
        return {"type": "face", "data": {"id": self.id}}


class Media:
    kind = "media"

    def __init__(self, file: str, name: str = ""):
        self.file, self.name = file, name

    @classmethod
    def fromFileSystem(cls, path: str) -> Media:
        return cls(Path(path).resolve().as_uri())

    async def to_dict(self) -> dict:
        return {"type": self.kind, "data": {"file": self.file, "name": self.name}}

    async def convert_to_base64(self) -> str:
        path = Path(unquote(urlparse(self.file).path))
        return base64.b64encode(path.read_bytes()).decode()


class Image(Media):
    kind = "image"


class Record(Media):
    kind = "record"


class Video(Media):
    kind = "video"


class File(Media):
    kind = "file"


class Node:
    def __init__(self, content: list[Any], uin: str, name: str, time: int = 0):
        self.content, self.uin, self.name, self.time = content, uin, name, time

    async def to_dict(self) -> dict:
        content = []
        for component in self.content:
            if isinstance(component, (Image, Record)):
                path = Path(unquote(urlparse(component.file).path))
                encoded = base64.b64encode(path.read_bytes()).decode()
                content.append(
                    {"type": component.kind, "data": {"file": "base64://" + encoded}}
                )
            else:
                content.append(await component.to_dict())
        return {
            "type": "node",
            "data": {
                "user_id": str(self.uin),
                "nickname": self.name,
                "content": content,
            },
        }


class Nodes:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

    async def to_dict(self) -> dict:
        return {"messages": [await node.to_dict() for node in self.nodes]}


COMPONENTS = types.ModuleType("astrbot.api.message_components")
for component in (Plain, Reply, At, Face, Image, Record, Video, File, Node, Nodes):
    setattr(COMPONENTS, component.__name__, component)


class FakeAPI:
    def __init__(self, handler=None):
        self.handler = handler
        self.calls: list[tuple[str, dict]] = []
        self.sent: list[Any] = []

    async def call_action(self, action: str, **params):
        self.calls.append((action, params))
        if action.startswith("send_"):
            self.sent.append(params.get("message", params.get("messages")))
            return {"message_id": len(self.sent)}
        if self.handler is None:
            raise RuntimeError("模拟接口不可用")
        result = self.handler(action, params)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, BaseException):
            raise result
        return result


class FakeEvent:
    def __init__(
        self,
        message=None,
        *,
        group="100",
        sender="12345",
        self_id="99999",
        platform="aiocqhttp",
        platform_id="qq-test",
        api=None,
    ):
        self.message = message or []
        self.group, self.sender, self.platform, self.platform_id = (
            group,
            sender,
            platform,
            platform_id,
        )
        self.message_obj = types.SimpleNamespace(self_id=self_id)
        self.bot = types.SimpleNamespace(api=api or FakeAPI())
        self.stopped = False
        self.sent: list[Any] = []

    def get_messages(self):
        return self.message

    def get_group_id(self):
        return self.group

    def get_sender_id(self):
        return self.sender

    def get_platform_name(self):
        return self.platform

    def get_platform_id(self):
        return self.platform_id

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text):
        return text

    def chain_result(self, chain):
        return chain

    async def send(self, result):
        self.sent.append(result)


def import_plugin_entry():
    import importlib

    api = types.ModuleType("astrbot.api")
    api.message_components = COMPONENTS
    api.AstrBotConfig = dict
    api.logger = logging.getLogger("submit-quotes-tests")
    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = FakeEvent
    event.filter = types.SimpleNamespace(
        EventMessageType=types.SimpleNamespace(ALL="all"),
        event_message_type=lambda *args, **kwargs: lambda function: function,
    )
    star = types.ModuleType("astrbot.api.star")

    class Star:
        def __init__(self, context):
            self.context = context

    star.Star, star.Context = Star, object
    modules = {
        "astrbot": types.ModuleType("astrbot"),
        "astrbot.api": api,
        "astrbot.api.message_components": COMPONENTS,
        "astrbot.api.event": event,
        "astrbot.api.star": star,
    }
    with patch.dict(sys.modules, modules):
        return importlib.import_module("astrbot_plugin_submit_quotes.main")
