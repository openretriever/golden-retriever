import copy


def parse_instruction(task, lang_goal, subtask=None):
    if task == "multi-language-conditioned":
        assert subtask is not None
        task = copy.copy(subtask)

    if "done" in lang_goal or "solve" in lang_goal:
        return lang_goal, lang_goal
    else:
        if "put-block-in-bowl" in task:
            pick_goal = " ".join(lang_goal.split(" ")[2:4])
            place_goal = " ".join(lang_goal.split(" ")[6:])
        elif "stack-block-pyramid-seq" in task:
            # "put the {pick} block on {place}"
            pp_split = lang_goal.split(" on ")
            pick_goal = " ".join(pp_split[0].split(" ")[2:])
            place_goal = " ".join(pp_split[1].split(" ")[1:])
        elif "assembling-kits-seq" in task:
            # "put the {color} {obj} in the {loc}{obj} hole"
            pp_split = lang_goal.split(" in ")
            pick_goal = " ".join(pp_split[0].split(" ")[2:])
            place_goal = " ".join(pp_split[1].split(" ")[1:])
        elif "google-objects-seq" in task:
            # "pack the {obj} in the brown box"
            pp_split = lang_goal.split(" in ")
            pick_goal = " ".join(pp_split[0].split(" ")[2:])
            place_goal = " ".join(pp_split[1].split(" ")[1:])
        elif "google-objects-seq" in task:
            # "pack the {obj} in the brown box"
            pp_split = lang_goal.split(" in ")
            pick_goal = " ".join(pp_split[0].split(" ")[2:])
            place_goal = " ".join(pp_split[1].split(" ")[1:])
        elif "google-objects-group" in task:
            # "pack all the {obj} objects in the brown box"
            pp_split = lang_goal.split(" in ")
            pick_goal = " ".join(pp_split[0].split(" ")[2:-1])
            place_goal = " ".join(pp_split[1].split(" ")[1:])
        elif "packing-boxes-pairs" in task:
            # "pack all the {colors} blocks into the brown box"
            pp_split = lang_goal.split(" into ")
            pick_goal = " ".join(pp_split[0].split(" ")[2:])
            place_goal = " ".join(pp_split[1].split(" ")[1:])
        elif "separating-piles" in task:
            # "push the pile of {block_color} blocks into the {square_color} square"
            pp_split = lang_goal.split(" into ")
            pick_goal = " ".join(pp_split[0].split(" ")[4:])
            place_goal = " ".join(pp_split[1].split(" ")[1:])
        elif "towers-of-hanoi" in task:
            # "move the {obj} ring to the {loc}"
            pp_split = lang_goal.split(" to ")
            pick_goal = " ".join(pp_split[0].split(" ")[2:])
            place_goal = " ".join(pp_split[1].split(" ")[1:])
        elif "align-rope" in task:
            # "align the rope from {direction}"
            pp_split = lang_goal.split(" from ")
            pick_goal = "rope"
            place_goal = " ".join(pp_split[1].split(" "))
        elif "packing-shapes" in task:
            # "pack the {obj} in the brown box"
            pp_split = lang_goal.split(" in ")
            pick_goal = " ".join(pp_split[0].split(" ")[2:])
            place_goal = " ".join(pp_split[1].split(" ")[1:])
        elif "put-block-in-box-real" in task:
            # pick: pick {color} block and place into the brown box
            pick_goal = " ".join(lang_goal.split(" ")[1:3])
            place_goal = "brown box"
        elif "pick-part-in-box-real" in task:
            # pick: pick {color} block and place into the brown box
            pp_split = lang_goal.split(" and ")
            pick_goal = " ".join(pp_split[0].split(" ")[1:])
            place_goal = "brown box"
        elif "processed" in task:
            if "pyramid" in task:
                # pick: pick {something} and place into {something}
                pick, place = lang_goal.split(" on ")
                pick_goal = " ".join(pick.split(" ")[1:-2])
                place_goal = " ".join(place.split(" ")[:])
            else:
                if lang_goal.count(" and ") == 2:
                    # temp solution for pyramid in multitask training

                    pick, place = lang_goal.split(" on ")
                    pick_goal = " ".join(pick.split(" ")[1:-2])
                    place_goal = " ".join(place.split(" ")[:])
                else:
                    if lang_goal == "pick navy teal block on black square":
                        lang_goal = "pick navy teal block and place on black square"
                    pick, place = lang_goal.split(" and ")
                    pick_goal = " ".join(pick.split(" ")[1:])
                    place_goal = " ".join(place.split(" ")[2:])
        else:
            NotImplementedError

        return pick_goal, place_goal
