class EnvironmentFactory:
    _backend = None  # Class-level variable to store the backend

    @classmethod
    def register(cls, backend):
        """
        Register a backend to the factory.
        :param backend: Backend class implementing `make` and `make_vec` methods.
        """
        cls._backend = backend

    @classmethod
    def make(cls, env_id, **kwargs):
        """
        Create an environment using the registered backend.
        :param env_id: Environment ID.
        :param kwargs: Additional arguments for environment creation.
        """
        if cls._backend is None:
            raise ValueError("No backend registered. Call 'EnvironmentFactory.register' first.")
        elif not hasattr(cls._backend, "make"):
            raise ValueError("Backend does not implement 'make' method.")
        return cls._backend.make(env_id, **kwargs)

    @classmethod
    def make_vec(cls, env_id, num_envs, **kwargs):
        """
        Create a vectorized environment using the registered backend.
        :param env_id: Environment ID.
        :param num_envs: Number of environments to vectorize.
        :param kwargs: Additional arguments for environment creation.
        """
        if cls._backend is None:
            raise ValueError("No backend registered. Call 'EnvironmentFactory.register' first.")
        elif not hasattr(cls._backend, "make_vec"):
            raise ValueError("Backend does not implement 'make_vec' method.")
        return cls._backend.make_vec(env_id, num_envs=num_envs, **kwargs)
