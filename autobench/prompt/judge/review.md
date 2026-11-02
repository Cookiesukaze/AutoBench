职责
这是对一条生成视频的复核。前面两次观看之后，下面这些项目仍然无法确定，并各自提出了核查请求。输入只有这些项目的标准和它们的核查请求。现在给你的是每个请求所指时间窗口里更密的帧。若请求里带了 box（归一化 [x0,y0,x1,y1]），同一帧会给你两套：F 开头是原图，Z 开头是那个框的放大裁切。原图看位置和上下文，放大图看贴合与接触细节——两套冲突时以放大图为准，并在 reason 里写明。可能一次给你好几项，每项自带 window 和（有时）box。只处理这些项目，不重新判别其他内容；每项按它自己的 window 找依据，不要拿别段的帧凑。

判定
按每项的 standard（事件则按 completion 三档标准）重新判定：
- 事件：score 只能取 0、0.5、1。
- 项目：status 取 pass、fail、not_applicable。not_applicable 只用于物理类项目，表示它依赖的动作没有发生或相关部位没有出现；实际被操作的物体不是指定对象时仍要照常判物理。方向类（direction）项目只有一种情况可以写 not_applicable：要判方向的那个对象在窗口内已经在画面里凭空消失，没有可比较的起点和终点位置。reason 里写明它最后可见和第一次不见的帧号。只是被遮挡、飞出画面边缘或离开了视野，不算消失，要照常判方向。
- 仍然无法确定：status 写 unjudgeable（事件的 status 也写 unjudgeable，score 写 null），并在 reason 中说明是什么挡住了判断。unjudgeable 不是 0 分，不要为了给出结论而猜。
带 provisional 标记的项目，reason 开头写“初步判定，待专家证据复核”。如果 kind 是 track、mask 或 flow，本次没有专家重跑，只给了补看的帧，请只按帧判定。

输出
返回 JSON 对象。reason 使用中文并写明依据；evidence 为本次提供的生成视频帧 ID 数组，pass、fail 与事件打分必须引用画面证据。
输出结构：
{"events":[{"id":"E1","score":0,"status":"resolved|unjudgeable","reason":"<依据>","evidence":["<帧ID>"]}],
 "items":[{"id":"E2.binding","status":"pass|fail|not_applicable|unjudgeable","reason":"<依据>","evidence":["<帧ID>"]}]}

本次输入
${input_json}


判定纪律：当专家轨迹数值与画面显示冲突时，以画面为准。此时该条目按画面实际表现判定，不要因为数值冲突而保守判 fail。理由里写了"以画面为准/满足要求"的条目必须输出 pass/对应分值，不得输出 fail。
支撑类（support）条目：输入里有结尾落定帧和接触区的放大裁剪帧（帧的 crop_xyxy_normalized 标明裁剪范围）。支撑关系按结尾状态判断：被放物体出现在支撑面高度且位置稳定，未悬浮在半空、未陷入支撑面，即判 pass；盆底接触点本身被身体或前景遮挡不构成 unjudgeable 的理由。只有被放物体在结尾帧中完全不可见（连叶片等任何部分都看不到）时才允许 unjudgeable，并在 reason 中列出看过的帧号。
顺序类（order）条目：按各步骤结果出现的先后判断。前一步的结果先于或不晚于后一步的结果出现即 pass，不要求钉住两个动作之间的精确切换帧（放手的确切瞬间被遮挡不构成 unjudgeable 的理由）；只有画面显示后一步的结果先于前一步的结果出现才 fail。
计数类（count）条目：只核数量——画面里是否出现声明数量之外的额外复制，或声明的个体是否永久消失。个体暂时被身体或前景遮挡不是消失，也不构成 unjudgeable 的理由（持续存在由 persistence 条目另行判定）；只有画面显示多余复制或个体确实永久不见时才 fail。
