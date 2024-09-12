import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'MorphDAG-prototype')))

from core.node import Node
from core.ttype.transaction import Transaction
from core.tp.state_transition import StateTransition
import json
import torch

class MorphDAGIntegration:
    def __init__(self, node_config):
        self.node = Node(node_config)
        self.node.start()

    def store_model(self, model_state_dict, client_id):
        model_bytes = self._serialize_model(model_state_dict)
        tx = Transaction(f"store_model_{client_id}", model_bytes)
        self.node.tp.add_tx(tx)

    def aggregate_models(self, client_ids):
        # Implement the logic to aggregate models using MorphDAG
        # This is a placeholder and needs to be implemented based on MorphDAG's specifics
        aggregated_model = None
        for client_id in client_ids:
            model_bytes = self._get_model_from_blockchain(client_id)
            model = self._deserialize_model(model_bytes)
            if aggregated_model is None:
                aggregated_model = model
            else:
                # Implement your aggregation logic here
                for key in aggregated_model.keys():
                    aggregated_model[key] += model[key]
        
        # Average the aggregated model
        for key in aggregated_model.keys():
            aggregated_model[key] /= len(client_ids)

        return aggregated_model

    def _serialize_model(self, model_state_dict):
        return json.dumps({key: value.tolist() for key, value in model_state_dict.items()})

    def _deserialize_model(self, model_bytes):
        model_dict = json.loads(model_bytes)
        return {key: torch.tensor(value) for key, value in model_dict.items()}

    def _get_model_from_blockchain(self, client_id):
        # Implement the logic to retrieve a model from MorphDAG
        # This is a placeholder and needs to be implemented based on MorphDAG's specifics
        pass

    def stop(self):
        self.node.stop()