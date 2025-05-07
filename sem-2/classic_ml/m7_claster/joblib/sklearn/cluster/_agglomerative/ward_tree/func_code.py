# first line: 182
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global_skip_validation = get_config()["skip_parameter_validation"]
            if global_skip_validation:
                return func(*args, **kwargs)

            func_sig = signature(func)

            # Map *args/**kwargs to the function signature
            params = func_sig.bind(*args, **kwargs)
            params.apply_defaults()

            # ignore self/cls and positional/keyword markers
            to_ignore = [
                p.name
                for p in func_sig.parameters.values()
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            to_ignore += ["self", "cls"]
            params = {k: v for k, v in params.arguments.items() if k not in to_ignore}

            validate_parameter_constraints(
                parameter_constraints, params, caller_name=func.__qualname__
            )

            try:
                with config_context(
                    skip_parameter_validation=(
                        prefer_skip_nested_validation or global_skip_validation
                    )
                ):
                    return func(*args, **kwargs)
            except InvalidParameterError as e:
                # When the function is just a wrapper around an estimator, we allow
                # the function to delegate validation to the estimator, but we replace
                # the name of the estimator by the name of the function in the error
                # message to avoid confusion.
                msg = re.sub(
                    r"parameter of \w+ must be",
                    f"parameter of {func.__qualname__} must be",
                    str(e),
                )
                raise InvalidParameterError(msg) from e
