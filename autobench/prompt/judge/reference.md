只检查所附参考图中的身份比对前提。本次没有生成视频，禁止推测生成质量、禁止按题面声称的人物存在来代替看图。
对每个 identity 项回答：
- applicable：参考图里确实出现该人物，存在可比较的外观依据。局部人物也可作为依据，不要求必须露出完整脸部。
- not_applicable：参考图明确没有这个人物或任何可供比对的人体部分。场景里的工具、把手、推测的画外操作者都不算可见人物。题面声称 in_picture 也不能制造参考人物。
- unjudgeable：图像无法读取、过暗、模糊或遮挡，不能确定人物是否存在；不得用 not_applicable 代替证据不足。
reason 用中文描述参考图实际可见的内容。此处只判有没有参考对象，不判视频身份是否保持，也不判通过或失败。
返回 JSON：{"items":[{"id":"G.identity","status":"applicable|not_applicable|unjudgeable","reason":"参考图中的视觉依据"}]}

${input_json}
