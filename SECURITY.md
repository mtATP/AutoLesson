# 安全说明

## 不要提交的内容

- `config.yaml` 中的真实 API Key；
- 课程 Cookie、账号密码、验证码；
- 真实姓名、学校、学号、班级；
- 个人课程页面 MHTML、作业文档、运行日志/截图。

仓库已通过 `.gitignore` 排除 `config.yaml`、`*.mhtml`、`runtime/`、`*.docx`、日志、备份和本地 IDE 文件。

## API Key 泄露

如果 Key 曾经出现在提交、截图、Issue 或聊天记录中：

1. 立即到模型平台撤销/轮换 Key；
2. 从配置文件删除；
3. 如果已推送到 GitHub，仅删除文件不够，还需要清理 Git 历史；
4. 检查账单和调用记录。

## 报告安全问题

请通过仓库维护者提供的私密渠道报告，不要在公开 Issue 中粘贴 Key、Cookie、姓名或课程数据。
