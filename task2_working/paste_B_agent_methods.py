# ---- Task 2: replace in `class Agent` ----
# 1) __init__ becomes:
#
#     def __init__(self):
#         self.actions = None
#         self.expanded = 0
#         self.model = load_model()
#
# 2) DELETE the old `split_cells` and `predict` methods, and replace `parse` with this one:

    def parse(self, observation: ImageObservation) -> GridState:
        """Convert an image observation into a GridState (also used by Grid Play to show the agent's view)."""
        info = observation["info"]
        config = info["config"]
        status = info["status"]
        width = config["width"]
        height = config["height"]

        gridstate = GridState(
            width=width,
            height=height,
            movement=MOVEMENTS["cardinal"],
            objective=OBJECTIVES["collect_gems_and_exit"],
            step_cost=STEP_COST,
            seed=None if config["seed"] == -1 else config["seed"],
            turn=status["turn"],
            score=status["score"],
            win=status["phase"] == "win",
            lose=status["phase"] == "lose",
            message=info["message"] or None,
            turn_limit=None if config["turn_limit"] == -1 else config["turn_limit"],
        )

        cells = classify_grid(observation["image"], width, height, self.model,
                              first_turn=(status["turn"] == 0))
        for k, names in enumerate(cells):
            x, y = k % width, k // width
            for name in names:
                entity = ENTITY_TYPES[name]()
                if isinstance(entity, AgentEntity):
                    entity.set_health(info["agent"]["health"]["current_health"])
                gridstate.add((x, y), entity)
        return gridstate
