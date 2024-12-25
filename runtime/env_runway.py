# env_runway.py - Refined inference API for clients.

from flask import Flask, request, jsonify
import logging
import uuid

HTTP_BAD_REQUEST = 400
HTTP_OK = 200

class EnvRunway:
    env_type = None  # The backend that handles model inference, sequences, and agent state

    def __init__(self):
        self.app = Flask(__name__)
        logging.basicConfig(level=logging.INFO)
        self._define_routes()

    def _define_routes(self):
        # Load the ONNX model that was previously trained using our training service
        self.app.add_url_rule("/load_model", "load_model", self.load_model, methods=["POST"])

        # Agent management for inference
        self.app.add_url_rule("/register_agent", "register_agent", self.register_agent, methods=["POST"])
        self.app.add_url_rule("/deregister_agent", "deregister_agent", self.deregister_agent, methods=["POST"])

        # Inference step: Get an action for a given agent and current observation
        self.app.add_url_rule("/select_action", "select_action", self.select_action, methods=["POST"])

        # Reset an agent’s internal state if needed (e.g., after an episode ends)
        self.app.add_url_rule("/reset_agent", "reset_agent", self.reset_agent, methods=["POST"])

        # Check the status of the model and agents
        self.app.add_url_rule("/status", "status", self.status, methods=["GET"])

    def load_model(self, model_path=None):
        """
        POST /load_model
        Request Body:
        {
          "model_path": "s3://bucket/path/to/model.onnx"
        }

        Loads the client's ONNX model for inference.
        The client sets this path to the location where the trained model is stored.
        This is typically the output of the training phase (e.g., after running sagemaker_server_test.py and EnvGateway).

        Response:
        {
          "message": "Model loaded successfully."
        }
        """
        data = request.json
        model_path = data.get("model_path")
        if model_path is None:
            return jsonify({"error": "Missing model_path"}), HTTP_BAD_REQUEST

        try:
            self.env_type.load_model(model_path)  # Backend should handle S3 download & ONNX init.
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            return jsonify({"error": "Failed to load model."}), HTTP_BAD_REQUEST

        return jsonify({"message": f"Model loaded from {model_path}."}), HTTP_OK

    def register_agent(self):
        """
        POST /register_agent
        Request Body:
        {
          "agent_id": "optional_custom_id",  // If omitted, a unique ID is generated.
          "initial_observation": [ ... ]     // The first observation for this agent.
        }

        Used when starting inference for a new agent. This sets up the sequence states internally.

        Response:
        {
          "agent_id": "assigned_or_provided_id",
          "message": "Agent registered successfully."
        }
        """
        data = request.json
        initial_observation = data.get("initial_observation")
        if initial_observation is None:
            return jsonify({"error": "Missing initial_observation"}), HTTP_BAD_REQUEST

        agent_id = data.get("agent_id")
        if agent_id is None:
            agent_id = str(uuid.uuid4())  # Generate a unique ID if not provided

        try:
            self.env_type.register_agent(agent_id, initial_observation)
        except Exception as e:
            logging.error(f"Failed to register agent {agent_id}: {e}")
            return jsonify({"error": f"Failed to register agent: {e}"}), HTTP_BAD_REQUEST

        return jsonify({"agent_id": agent_id, "message": "Agent registered successfully."}), HTTP_OK

    def deregister_agent(self):
        """
        POST /deregister_agent
        Request Body:
        {
          "agent_id": "agent_id_to_remove"
        }

        Removes the agent and clears its internal state from the system.

        Response:
        {
          "message": "Agent <id> deregistered successfully."
        }
        """
        data = request.json
        agent_id = data.get("agent_id")
        if agent_id is None:
            return jsonify({"error": "Missing agent_id"}), HTTP_BAD_REQUEST

        try:
            self.env_type.deregister_agent(agent_id)
        except Exception as e:
            logging.error(f"Failed to deregister agent {agent_id}: {e}")
            return jsonify({"error": f"Failed to deregister agent: {e}"}), HTTP_BAD_REQUEST

        return jsonify({"message": f"Agent {agent_id} deregistered successfully."}), HTTP_OK

    def select_action(self):
        """
        POST /select_action
        Request Body:
        {
          "agent_id": "agent_id",
          "observation": [ ... ]  // Current observation for this time step
        }

        The backend uses the agent’s current observation sequence, updates it with the provided observation,
        and runs inference to produce an action.

        Response:
        {
          "action": [ ... ],
          "info": { ... } // Optional additional info
        }
        """
        data = request.json
        agent_id = data.get("agent_id")
        observation = data.get("observation")

        if agent_id is None or observation is None:
            return jsonify({"error": "Missing agent_id or observation"}), HTTP_BAD_REQUEST

        try:
            action, info = self.env_type.select_action(agent_id, observation)
        except Exception as e:
            logging.error(f"Failed to select action for agent {agent_id}: {e}")
            return jsonify({"error": f"Failed to select action: {e}"}), HTTP_BAD_REQUEST

        return jsonify({"action": action, "info": info}), HTTP_OK

    def reset_agent(self):
        """
        POST /reset_agent
        Request Body:
        {
          "agent_id": "agent_id"
        }

        Resets the agent’s internal state (e.g., if the environment episode ended and you want to start a new episode).

        Response:
        {
          "message": "Agent <id> reset successfully."
        }
        """
        data = request.json
        agent_id = data.get("agent_id")
        if agent_id is None:
            return jsonify({"error": "Missing agent_id"}), HTTP_BAD_REQUEST

        try:
            self.env_type.reset_agent(agent_id)
        except Exception as e:
            logging.error(f"Failed to reset agent {agent_id}: {e}")
            return jsonify({"error": f"Failed to reset agent: {e}"}), HTTP_BAD_REQUEST

        return jsonify({"message": f"Agent {agent_id} reset successfully."}), HTTP_OK

    def status(self):
        """
        GET /status

        Returns the current status of the inference system.

        Response:
        {
          "model_loaded": bool,
          "num_agents": int,
          "details": { ... }
        }

        Clients can call this to check if the model is loaded and how many agents are currently registered.
        """
        try:
            model_loaded = self.env_type.is_model_loaded()
            num_agents = self.env_type.get_num_agents()
            details = self.env_type.get_status_details()  # Could return additional info
        except Exception as e:
            logging.error(f"Failed to retrieve status: {e}")
            return jsonify({"error": "Failed to retrieve status."}), HTTP_BAD_REQUEST

        status_info = {
            "model_loaded": model_loaded,
            "num_agents": num_agents,
            "details": details
        }

        return jsonify(status_info), HTTP_OK

    @classmethod
    def run(cls, backend, port=5001, host="0.0.0.0"):
        """
        Start the EnvRunway server for inference.
        The backend is a class or instance that implements the following methods:
            - load_model(model_path)
            - register_agent(agent_id, initial_observation)
            - deregister_agent(agent_id)
            - select_action(agent_id, observation) -> (action, info)
            - reset_agent(agent_id)
            - is_model_loaded() -> bool
            - get_num_agents() -> int
            - get_status_details() -> dict (additional status info)

        :param backend: The backend object handling the inference logic.
        :param port: Server port.
        :param host: Host address.
        """
        cls.env_type = backend
        logging.info(f"Backend {backend.__name__} registered for inference.")
        runway = cls()
        runway.app.run(host=host, port=port)
