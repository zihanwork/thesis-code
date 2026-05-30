"""VirtualHome task-level sequence templates for the knowledge graph.

Direction 1: beyond single-step preconditions, these templates encode
common multi-step task patterns (e.g. laundry, cooking, hygiene).
Each template is a named ordered sequence of (action, arg_class_hint)
steps that must occur in order for the task to succeed.

Templates are loaded into Neo4j as TaskTemplate nodes linked via
STEP_OF edges to Action nodes, with ordering preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StepSpec:
    action: str                          # VirtualHome action name
    arg1_class: Optional[str] = None     # expected object class for arg1
    arg2_class: Optional[str] = None     # expected object class for arg2
    note: str = ""


@dataclass
class TaskTemplate:
    name: str
    category: str
    description: str
    steps: List[StepSpec] = field(default_factory=list)
    # Objects the task typically involves (used for RAG seed boosting)
    key_objects: List[str] = field(default_factory=list)


TASK_TEMPLATES: Dict[str, TaskTemplate] = {
    # ---------------------------------------------------------------- laundry
    "laundry_wash": TaskTemplate(
        name="laundry_wash",
        category="cleaning",
        description="Put clothes into washing machine and start it",
        key_objects=["washing_machine", "clothes_jacket", "clothes_pants", "clothes_shirt"],
        steps=[
            StepSpec("WALK", arg1_class="washing_machine"),
            StepSpec("OPEN", arg1_class="washing_machine"),
            StepSpec("GRAB", arg1_class="clothes", note="repeat for each clothing item"),
            StepSpec("PUTIN", arg1_class="clothes", arg2_class="washing_machine"),
            StepSpec("CLOSE", arg1_class="washing_machine"),
            StepSpec("SWITCHON", arg1_class="washing_machine"),
        ],
    ),
    "laundry_with_detergent": TaskTemplate(
        name="laundry_with_detergent",
        category="cleaning",
        description="Add detergent then wash clothes",
        key_objects=["washing_machine", "laundry_detergent", "clothes_jacket"],
        steps=[
            StepSpec("WALK", arg1_class="laundry_detergent"),
            StepSpec("GRAB", arg1_class="laundry_detergent"),
            StepSpec("WALK", arg1_class="washing_machine"),
            StepSpec("OPEN", arg1_class="washing_machine"),
            StepSpec("PUTIN", arg1_class="laundry_detergent", arg2_class="washing_machine"),
            StepSpec("GRAB", arg1_class="clothes", note="repeat for each item"),
            StepSpec("PUTIN", arg1_class="clothes", arg2_class="washing_machine"),
            StepSpec("CLOSE", arg1_class="washing_machine"),
            StepSpec("SWITCHON", arg1_class="washing_machine"),
        ],
    ),
    # ---------------------------------------------------------------- cooking
    "microwave_food": TaskTemplate(
        name="microwave_food",
        category="cooking",
        description="Heat food in microwave",
        key_objects=["microwave", "plate", "food"],
        steps=[
            StepSpec("WALK", arg1_class="food_item"),
            StepSpec("GRAB", arg1_class="food_item"),
            StepSpec("WALK", arg1_class="microwave"),
            StepSpec("OPEN", arg1_class="microwave"),
            StepSpec("PUTIN", arg1_class="food_item", arg2_class="microwave"),
            StepSpec("CLOSE", arg1_class="microwave"),
            StepSpec("SWITCHON", arg1_class="microwave"),
        ],
    ),
    "boil_water": TaskTemplate(
        name="boil_water",
        category="cooking",
        description="Boil water on stove",
        key_objects=["stove", "pot", "water"],
        steps=[
            StepSpec("WALK", arg1_class="pot"),
            StepSpec("GRAB", arg1_class="pot"),
            StepSpec("WALK", arg1_class="stove"),
            StepSpec("PUTBACK", arg1_class="pot", arg2_class="stove"),
            StepSpec("SWITCHON", arg1_class="stove"),
        ],
    ),
    # ---------------------------------------------------------------- hygiene
    "brush_teeth": TaskTemplate(
        name="brush_teeth",
        category="hygiene",
        description="Brush teeth with toothbrush and toothpaste",
        key_objects=["toothbrush", "toothpaste", "sink"],
        steps=[
            StepSpec("WALK", arg1_class="toothbrush"),
            StepSpec("GRAB", arg1_class="toothbrush"),
            StepSpec("WALK", arg1_class="toothpaste"),
            StepSpec("GRAB", arg1_class="toothpaste"),
            StepSpec("WALK", arg1_class="sink"),
        ],
    ),
    "shower": TaskTemplate(
        name="shower",
        category="hygiene",
        description="Take a shower",
        key_objects=["shower", "towel"],
        steps=[
            StepSpec("WALK", arg1_class="shower"),
            StepSpec("SWITCHON", arg1_class="shower"),
            StepSpec("SWITCHOFF", arg1_class="shower"),
            StepSpec("WALK", arg1_class="towel"),
            StepSpec("GRAB", arg1_class="towel"),
        ],
    ),
    # ---------------------------------------------------------------- food & drink
    "eat_meal": TaskTemplate(
        name="eat_meal",
        category="eating",
        description="Pick up food and eat it",
        key_objects=["plate", "food"],
        steps=[
            StepSpec("WALK", arg1_class="food_item"),
            StepSpec("GRAB", arg1_class="food_item"),
            StepSpec("EAT", arg1_class="food_item"),
        ],
    ),
    "drink_beverage": TaskTemplate(
        name="drink_beverage",
        category="eating",
        description="Pick up a drink and drink it",
        key_objects=["glass", "cup", "beverage"],
        steps=[
            StepSpec("WALK", arg1_class="drink_item"),
            StepSpec("GRAB", arg1_class="drink_item"),
            StepSpec("DRINK", arg1_class="drink_item"),
        ],
    ),
    # ---------------------------------------------------------------- storage
    "put_away_item": TaskTemplate(
        name="put_away_item",
        category="storage",
        description="Pick up item and put it in a container",
        key_objects=["cabinet", "fridge", "box"],
        steps=[
            StepSpec("WALK", arg1_class="item"),
            StepSpec("GRAB", arg1_class="item"),
            StepSpec("WALK", arg1_class="container"),
            StepSpec("OPEN", arg1_class="container"),
            StepSpec("PUTIN", arg1_class="item", arg2_class="container"),
            StepSpec("CLOSE", arg1_class="container"),
        ],
    ),
    "retrieve_item": TaskTemplate(
        name="retrieve_item",
        category="storage",
        description="Open container, grab item, close container",
        key_objects=["fridge", "cabinet"],
        steps=[
            StepSpec("WALK", arg1_class="container"),
            StepSpec("OPEN", arg1_class="container"),
            StepSpec("GRAB", arg1_class="item"),
            StepSpec("CLOSE", arg1_class="container"),
        ],
    ),
    # ---------------------------------------------------------------- relaxation
    "watch_tv": TaskTemplate(
        name="watch_tv",
        category="leisure",
        description="Turn on TV and sit to watch",
        key_objects=["tv", "sofa", "remote_control"],
        steps=[
            StepSpec("WALK", arg1_class="tv"),
            StepSpec("SWITCHON", arg1_class="tv"),
            StepSpec("WALK", arg1_class="sofa"),
            StepSpec("SIT", arg1_class="sofa"),
            StepSpec("WATCH", arg1_class="tv"),
        ],
    ),
    "read_book": TaskTemplate(
        name="read_book",
        category="leisure",
        description="Pick up book and read it",
        key_objects=["book"],
        steps=[
            StepSpec("WALK", arg1_class="book"),
            StepSpec("GRAB", arg1_class="book"),
            StepSpec("READ", arg1_class="book"),
        ],
    ),
}
