"""Serialize nested forwards without losing adjacent text on NapCat."""

from __future__ import annotations

from typing import Any


def make_replay_nodes_type(components: Any) -> type:
    class ReplayNodes(components.Nodes):
        async def to_dict(self) -> dict:
            messages: list[dict] = []
            for node in self.nodes:
                ordinary: list[Any] = []

                async def flush() -> None:
                    if ordinary:
                        native = components.Node(
                            uin=node.uin,
                            name=node.name,
                            time=node.time,
                            content=list(ordinary),
                        )
                        payload = await native.to_dict()
                        for index, component in enumerate(ordinary):
                            if isinstance(component, components.Video):
                                payload["data"]["content"][
                                    index
                                ] = await component.to_dict()
                        payload["data"]["time"] = node.time
                        messages.append(payload)
                        ordinary.clear()

                for segment in node.content:
                    if isinstance(segment, ReplayNodes):
                        await flush()
                        nested = await segment.to_dict()
                        # NapCat 的嵌套节点分支会丢弃同节点内的其他消息段。
                        # 单独包裹内层转发，保留前后内容、顺序和作者信息。
                        messages.append(
                            {
                                "type": "node",
                                "data": {
                                    "user_id": str(node.uin),
                                    "nickname": node.name,
                                    "time": node.time,
                                    "content": nested["messages"],
                                },
                            }
                        )
                    else:
                        ordinary.append(segment)
                await flush()
            return {"messages": messages}

    return ReplayNodes
