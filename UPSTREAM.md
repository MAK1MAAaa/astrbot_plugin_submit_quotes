# 上游来源

- 仓库：[exynos967/astrbot_quotes_plugin](https://github.com/exynos967/astrbot_quotes_plugin)
- 作者：薄暝
- 采用版本：v2.6.0
- 固定提交：[feb4bb3df153c8e9b20ab9c0c8ebe1ef6085d4f5](https://github.com/exynos967/astrbot_quotes_plugin/commit/feb4bb3df153c8e9b20ab9c0c8ebe1ef6085d4f5)
- 查阅日期：2026-10-10
- 适配作者：makima

源码以该提交为基础，复用并改造 `napcat_service.py` 的 OneBot 消息／转发读取、`models.py` 的语录与媒体模型、`utils.py` 的 CQ 解析／媒体准备／数据目录定位，以及上游按会话保存语录和媒体的目录结构。依赖沿用 `httpx`、`Pillow`。

本版本将入口改为精确监听“投稿”和“语录”，普通消息扩展至与转发节点相同的媒体类型，移除跳过内容和占位符降级；加入下载限制、来源去重、全量成功后确认、锁内原子快照、失败资源回滚，以及三层转发的兼容回放。`message_service.py`、`store.py`、`quote_service.py` 已按这些行为重写或扩展，新增 `replay.py` 和离线行为测试。

不包含上游的语录卡片渲染、搜索、删除或其他命令，也不将上游仓库地址当作本插件的发布地址。

所采用提交的源码包未包含 `LICENSE` 文件。本目录保留上游作者和来源记录，未自行为上游代码指定开源许可证。
