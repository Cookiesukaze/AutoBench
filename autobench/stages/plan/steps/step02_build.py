"""事件时间轴与每个小项的判定标准。

文本生成规则与旧仓 plan.py 逐字一致：同一份 spec 必须读出同一句话，
否则判定标准的措辞会在移植中悄悄变化。
"""

from __future__ import annotations


from ....app.data.types import ContractError, Event, Item, Plan

# 镜头文案需要这几个表；值来自 taskdesign/spec.py，原样搬。
MOTIONS = {
    "static": "Static Shot",
    "pan_left": "Pan Left", "pan_right": "Pan Right",
    "tilt_up": "Tilt Up", "tilt_down": "Tilt Down",
    "zoom_in": "Zoom In", "zoom_out": "Zoom Out",
    "push_in": "Push In", "pull_out": "Pull Out",
    "truck_left": "Truck Left", "truck_right": "Truck Right",
    "pedestal_up": "Pedestal Up", "pedestal_down": "Pedestal Down",
    "arc": "Arc Shot", "tracking": "Tracking Shot",
}
MOVING_POSITION = {"push_in", "pull_out", "truck_left", "truck_right",
                   "pedestal_up", "pedestal_down", "arc", "tracking"}
CAMERA_VERB = {
    "pan_left": "pans left", "pan_right": "pans right",
    "tilt_up": "tilts up", "tilt_down": "tilts down",
    "zoom_in": "zooms in", "zoom_out": "zooms out",
    "push_in": "pushes in", "pull_out": "pulls out",
    "truck_left": "trucks left", "truck_right": "trucks right",
    "pedestal_up": "pedestals up", "pedestal_down": "pedestals down",
}
ARTICLES = ("the ", "a ", "an ", "her ", "his ", "their ", "its ", "both ", "this ", "that ")
SUPPORT_RELATIONS = ("on", "in", "into", "onto")
BINARY = (0.0, 1.0)


# 文本 ------------------------------------------------------------------

def lower(text: str) -> str:
    text = text.strip().rstrip(".")
    return text[:1].lower() + text[1:] if text and not text[:2].isupper() else text


def with_article(noun: str) -> str:
    noun = noun.strip()
    return noun if noun.lower().startswith(ARTICLES) else "the " + noun


def sentence(text: str) -> str:
    text = text.strip().rstrip(".")
    if text and text[0].islower():
        text = text[:1].upper() + text[1:]
    return text + "."


def resolve(ref: str, subjects: list[dict]) -> str:
    for subject in subjects:
        if subject["id"] == ref:
            return subject_phrase(subject)
    return ref


def subject_phrase(subject: dict) -> str:
    description = subject["description"].strip().rstrip(".")
    person = subject.get("person_type")
    return f"{person} {description}" if person else description


def short(ref: str, subjects: list[dict]) -> str:
    for subject in subjects:
        if subject["id"] == ref:
            person = subject.get("person_type")
            twins = [s for s in subjects if s.get("person_type") == person and person]
            if person and len(twins) == 1:
                return person
            return subject_phrase(subject) if person else subject["description"].strip()
    return ref


def clause(segment: dict, subjects: list[dict]) -> str:
    actor = resolve(segment["actor"], subjects)
    part = f'{with_article(actor)} {segment["action"].strip()}'
    obj = resolve(segment["object"], subjects) if segment["object"] else ""
    if obj and not part.lower().endswith(obj.lower()):
        part += " " + with_article(obj)
    if segment.get("destination"):
        part += f' {segment["destination_relation"]} {with_article(resolve(segment["destination"], subjects))}'
    hand = {"left": "with the left hand", "right": "with the right hand", "both": "with both hands"}
    if segment["hand"] in hand and f'{segment["hand"]} hand' not in part.lower():
        part += " " + hand[segment["hand"]]
    if segment.get("instrument"):
        part += f' using {with_article(resolve(segment["instrument"], subjects))}'
    return part


def camera_params(camera: dict) -> dict:
    pan = tilt = zoom = "none"
    position = "any" if camera["motion"] in MOVING_POSITION else "fixed"
    mapping = {"pan_left": ("pan", "left"), "pan_right": ("pan", "right"),
               "tilt_up": ("tilt", "up"), "tilt_down": ("tilt", "down"),
               "zoom_in": ("zoom", "in"), "zoom_out": ("zoom", "out")}
    if camera["motion"] in mapping:
        key, value = mapping[camera["motion"]]
        if key == "pan":
            pan = value
        elif key == "tilt":
            tilt = value
        else:
            zoom = value
    return {"pan": pan, "tilt": tilt, "zoom": zoom, "position": position}


def camera_phrase(camera: dict, subjects: list[dict]) -> str:
    motion = camera["motion"]
    target = resolve(camera["target"], subjects) if camera["target"] else None
    if motion == "static":
        return f'The camera holds a static shot on {target or "the subject"}.'
    if motion == "arc":
        phrase = "The camera arcs around the subject"
    elif motion == "tracking":
        phrase = f"The camera follows {target} in a tracking shot"
    else:
        phrase = "The camera " + CAMERA_VERB[motion]
    if camera["amplitude"] != "medium":
        phrase += f' with {camera["amplitude"]} amplitude'
    if camera["speed"] != "medium":
        phrase += f' at {camera["speed"]} speed'
    if target and motion in ("push_in", "pull_out", "zoom_in", "zoom_out"):
        phrase += f" on {with_article(target)}"
    return phrase + "."


def end_state_text(spec: dict) -> str:
    if spec["end_frame"]:
        return spec["end_frame"]["detail"].strip().rstrip(".")
    sustained = [segment for segment in spec["segments"] if segment["sustain"]]
    segment = sustained[-1] if sustained else spec["segments"][-1]
    return segment["result"].strip().rstrip(".")


def concept(subject: dict) -> str:
    """SAM 拿到的那个词：人用 person_type，物用 segment_word。没有就报错，不猜。"""
    if subject.get("person_type"):
        return subject["person_type"]
    if subject.get("segment_word"):
        return subject["segment_word"]
    raise ContractError(f'subject {subject["id"]} 既没有 person_type 也没有 segment_word')


# 构造 ------------------------------------------------------------------

def _step(judged_by: str, expert: str | None) -> str:
    """规则自己判；要专家证据的项等融合阶段再问；其余首轮就判。"""
    if judged_by == "rule":
        return "expert"
    return "fusion" if expert else "first"


def item(checks: dict, item_id: str, type_name: str, standard: str,
         concepts: tuple[str, ...] = (), params: dict | None = None,
         levels: tuple[float, ...] = BINARY) -> Item:
    spec = checks["checks"].get(type_name)
    if spec is None:
        raise ContractError(f"checks.json 里没有检查项类型 {type_name!r}")
    return Item(
        id=item_id, type=type_name, dimension=spec["dimension"],
        judged_by=spec["judged_by"], expert=spec.get("expert"),
        standard=standard, rule=dict(spec.get("rule") or {}),
        levels=tuple(spec.get("levels") or levels),
        params=dict(params or {}), concepts=tuple(concepts),
        step=_step(spec["judged_by"], spec.get("expert")),
    )


def build(spec: dict, question_id: str, image_sha256: str, checks: dict) -> Plan:
    subjects = spec["subjects"]
    by_id = {s["id"]: s for s in subjects}
    segments = spec["segments"]
    events: list[Event] = []

    for index, segment in enumerate(segments, 1):
        eid = f"E{index}"
        actor = resolve(segment["actor"], subjects)
        obj = resolve(segment["object"], subjects) if segment["object"] else None
        actor_s = by_id[segment["actor"]]
        concepts = [concept(actor_s)]
        if segment["object"]:
            concepts.append(concept(by_id[segment["object"]]))
        body = clause(segment, subjects)
        sustain = " The result holds to the end of the step." if segment["sustain"] else ""
        completion = {
            "full": sentence(f'{body}, so that {lower(segment["result"])}') + sustain,
            "partial": "The actor visibly starts this step toward the right target, but the result is not "
                       "reached" + (" or is not held to the end of the step." if segment["sustain"] else ".") +
                       (" Also partial: the whole step is done, but on a thing of the same kind as the named "
                        f"object that looks different from it (another kind of {obj}); an object of a "
                        "different kind scores none." if obj else ""),
            "none": "The step does not happen, or something else happens instead.",
        }

        constraints: list[Item] = []
        physics: list[Item] = []
        if segment["hand"] != "none":
            constraints.append(item(
                checks, f"{eid}.hand_side", "hand_side",
                "The step is done with " + ("both hands." if segment["hand"] == "both"
                                            else f'the {segment["hand"]} hand.') +
                " Fail only when the wrong hand is used (a left hand where the prompt asks for a "
                "right one), and only when the pose keypoints agree with that reading: at this video "
                "resolution a hand seen at an angle is often misread, so when your view of the hand "
                "conflicts with the left/right wrist keypoint measurements, keep it open "
                "(needs_review) instead of failing. One hand doing the work while the other assists "
                "off-frame or rests is not a fail.",
                tuple(concepts),
                {"actor": actor, "hand": segment["hand"], "object": obj or actor}))
        if "binding" in segment["tests"] and obj:
            constraints.append(item(
                checks, f"{eid}.binding", "binding",
                f"It is {actor} who does this step, not another person. Whether the object is the "
                "right one is scored in the step itself and is not counted again here.",
                tuple(concepts), {"actor": actor, "object": obj}))
        if "direction" in segment["tests"] and segment["destination"]:
            where = resolve(segment["destination"], subjects)
            constraints.append(item(
                checks, f"{eid}.direction", "direction",
                f"{obj or actor} moves toward {where} in this step. Judge the net displacement from "
                "the step start to its end: wobbling or circling on the way is not a fail as long as "
                "the subject ends the step closer to or arriving at " + where + ". Fail only when the "
                "net displacement points away from " + where + "; a step that never reaches its "
                "destination is scored by the event completion, not here.",
                tuple(concepts), {"subject": obj or actor, "direction": f"toward {where}"}))

        if actor_s["kind"] == "person":
            physics.append(item(
                checks, f"{eid}.body", "body",
                f"The body of {actor} keeps a plausible structure during this step.",
                tuple(concepts[:1]), {"subject": actor}))
        if segment["object"]:
            physics.append(item(
                checks, f"{eid}.persistence", "persistence",
                "The object handled in this step does not vanish, split or merge during it. "
                "An object being eaten, drunk or peeled away is being consumed normally, not vanishing.",
                tuple(concepts[1:]),
                {"objects": [obj], "actor": actor,
                 "actor_concept": concept(by_id[segment["actor"]])}))
            if segment["hand"] != "none":
                physics.append(item(
                    checks, f"{eid}.contact", "contact",
                    "Where the hand holds or touches the object it handles in this step, the hold "
                    "stays unbroken and solid things do not pass through each other.",
                    tuple(concepts), {}))
        if segment["destination"] and segment["destination_relation"] in SUPPORT_RELATIONS:
            where = resolve(segment["destination"], subjects)
            support_concepts = []
            if segment["object"] and segment["object"] != segment["destination"]:
                support_concepts.append(concept(by_id[segment["object"]]))
            support_concepts.append(concept(by_id[segment["destination"]]))
            physics.append(item(
                checks, f"{eid}.support", "support",
                f'What is placed {segment["destination_relation"]} {where} rests on it and neither '
                "floats nor sinks into it. Fail only after the put-down has happened inside this "
                "step; while the actor still carries the object the step is simply not finished yet, "
                "which the event completion scores, not this item. A liquid surface (water and the "
                "like) is not a support to judge: objects floating or moving on it are normal.",
                tuple(support_concepts), {}))
        if "material" in segment["tests"] and obj:
            physics.append(item(
                checks, f"{eid}.material", "material",
                f"The material of {obj} changes in this step exactly as the task declares: the "
                "before-state and the after-state are both visible (the substance it turns into can "
                "be seen, not inferred), the change happens through a visible process, and nothing "
                f"else about {obj} changes with it.",
                tuple(concepts[1:]), {"object": obj}))
        if segment["destination"]:
            target = by_id[segment["destination"]]
            if target["kind"] in ("person", "animal") and target["id"] != segment["actor"]:
                physics.append(item(
                    checks, f"{eid}.interaction", "interaction",
                    f'{actor} affects {target["id"]} through visible contact only: the thing that '
                    "moves (the target, or an object it holds) moves where the actor or the handled "
                    "object touches it, nothing moves by intention or from a distance, and the two "
                    "remain separate bodies.",
                    tuple(concepts), {"actor": actor, "target": target["id"]}))

        events.append(Event(id=eid, what=body, completion=completion,
                            dimension=checks["checks"]["action"]["dimension"],
                            constraints=tuple(constraints), physics=tuple(physics),
                            concepts=tuple(concepts)))

    whole: list[Item] = []
    first = subjects[0]
    if first["kind"] == "person" and first["source"] == "in_picture":
        main = first["description"].strip().rstrip(".")
        whole.append(item(checks, "G.identity", "identity",
                          f"{main} keeps the identity and appearance it has in the reference picture.",
                          (concept(first),), {"subject": main}))

    ordered = [f"E{i}" for i, segment in enumerate(segments, 1)
               if not any(o is not segment and o["start"] < segment["end"] and segment["start"] < o["end"]
                          for o in segments)]
    if len(ordered) > 1:
        whole.append(item(
            checks, "G.order", "order",
            "The steps that do happen follow the order " + ", then ".join(ordered) +
            ". Only the order is judged here; whether a step used the right object or hand is "
            "scored in that step and is not counted again.",
            (), {"before": ordered[0], "after": ordered[-1]}))

    whole.append(item(
        checks, "G.camera", "camera",
        sentence(camera_phrase(spec["camera"], subjects)) +
        " Judge this movement; the camera may also move toward or away from the subject when that "
        "is what is asked. A camera that follows the moving subject and keeps it framed is "
        "performing the requested pan as well: reframing to follow an actor satisfies a slow-pan "
        "request and is not a fail. Sign convention: when the scene content slides LEFT across the "
        "frame, the camera is panning RIGHT, and vice versa; judge the pan direction by which way "
        "the content slides.",
        (), camera_params(spec["camera"])))

    mover = resolve(segments[0]["actor"], subjects)
    whole.append(item(
        checks, "G.continuity", "continuity",
        f"{mover} is one continuous body: no point of it jumps from frame to frame.",
        (concept(by_id[segments[0]["actor"]]),), {"subject": mover}))

    whole.append(item(
        checks, "G.end_state", "end_state",
        f"The final second shows {lower(end_state_text(spec))}. Score it on three levels: 1 the "
        "requested end state is fully there, 0.9 the video is heading there (the requested action "
        "is under way at the end) even if it has not fully landed, 0.5 the requested end state is "
        "nowhere in sight. Only these values are valid; do not use pass or fail for this item.",
        (), {"state": end_state_text(spec)}, levels=(1.0, 0.9, 0.5)))

    all_concepts = list(dict.fromkeys(concept(s) for s in subjects
                                      if s.get("kind") in ("animal", "person")))
    if all_concepts:
        whole.append(item(
            checks, "G.persistence", "persistence",
            "Every person or animal stays one continuous body: none of them vanishes, splits, "
            "merges or changes to an implausible size at any point.",
            tuple(all_concepts),
            {"objects": [s["id"] for s in subjects if s.get("kind") in ("animal", "person")]}))

    handled = list(dict.fromkeys(seg["object"] for seg in segments
                                 if seg.get("object") and by_id[seg["object"]]["kind"] in ("object", "vehicle")))
    prop_concepts = list(dict.fromkeys(concept(by_id[i]) for i in handled))
    if prop_concepts:
        whole.append(item(
            checks, "G.persistence.props", "persistence",
            "Every object a step handled stays present to the end of the video unless the task "
            "itself consumes it: eating, drinking, peeling or a stated consumption is normal, but "
            "an unexplained disappearance (it is there, then the mouth or hands are empty and it is "
            "nowhere) is not.",
            tuple(prop_concepts), {"objects": handled}))

    for side in subjects:
        if side.get("attribute"):
            whole.append(item(
                checks, f'G.attribute.{side["id"]}', "attribute",
                f'{side["id"]} keeps the attribute the task declares: {side["attribute"]}. Judge the '
                "declared attribute only; anything else about the subject is not this item.",
                (concept(side),), {"subject": side["id"], "attribute": side["attribute"]}))
    for side in subjects:
        if side["kind"] not in ("object", "vehicle"):
            continue
        declared = (f' The prompt itself brings this {side["id"]} into the scene: the one the actor '
                    "carries in is the declared copy, not an extra one; only a further copy beyond "
                    "the declared one fails this item.") if side.get("source") == "enters_frame" else ""
        whole.append(item(
            checks, f'G.count.{side["id"]}', "count",
            f'{side["id"]} appears in the video exactly as many times as it exists in the reference '
            "picture and the prompt: no extra copy of it ever shows up, and a copy that is present "
            "never disappears for good." + declared,
            (concept(side),), {"object": side["id"], "number": "declared"}))

    return Plan(
        question_id=question_id,
        prompt=spec["prompt_en"],
        reference_sha256=image_sha256,
        events=tuple(events),
        items=tuple(whole),
        subject_descriptions={concept(s): (s.get("description") or "").strip() for s in subjects},
    )
