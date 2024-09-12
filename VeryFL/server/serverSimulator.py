from server.base.baseAggregator import ServerAggregator
from server.morphdag_integration import MorphDAGIntegration
import os
import torch
from copy import deepcopy
from typing import OrderedDict

class serverSimulator:
    def __init__(
        self,
        aggregator: ServerAggregator,
        client_num=10,
        args=None
    ) -> None:
        self.aggregator = aggregator
        self.client_num = client_num
        self.global_model = None
        if args is not None:
            self.args = args
        self.morphdag = MorphDAGIntegration(args.get('node_config', {}))
        self.upload_model_list = []

    def _clear_upload_model_list(self):
        self.upload_model_list = []
        
    def _is_all_client_upload(self) -> bool:
        return len(self.upload_model_list) >= self.client_num

    def _set_global_model(self, global_model):
        self.global_model = global_model

    def _set_test_dataset(self, test_dataset):
        self.test_dataset = test_dataset
        self.test_batch_size = test_dataset.batch_size
        if test_dataset is None:
            raise Exception("Need to provide test dataset.")
        
    def _load_model(self):
        save_path = str(self.args['checkpoint_folder'])
        file_name = str(self.args['model'])
        file_path = save_path + file_name
        self.global_model.load_state_dict(torch.load(file_path))
        return

    def save_model(self, file_name='saved_model'):
        save_path = str(self.args['checkpoint_folder'])
        file_path_prefix = save_path + file_name
        if not os.path.isfile(file_path_prefix): 
            torch.save(self.global_model.state_dict(), file_path_prefix)
        else:
            count = 0
            file_path = file_path_prefix + str(count)
            while os.path.isfile(file_path):
                count = count + 1
                file_path = file_path_prefix + str(count)
            torch.save(self.global_model.state_dict(), file_path)
        return True
    
    def upload_model(self, upload_params: dict):
        client_id = upload_params['client_id']
        model_state_dict = upload_params['state_dict']
        self.morphdag.store_model(model_state_dict, client_id)
        self.upload_model_list.append(client_id)
        if self._is_all_client_upload():
            trained_model = self.morphdag.aggregate_models(self.upload_model_list)
            self._set_global_model(trained_model)
            self._clear_upload_model_list()
        
    def download_model(self, params=None) -> OrderedDict:
        if self.global_model is None:
            return "Failed to get global model"
        else:
            return deepcopy(self.global_model)
    
    def test(self):
        pass
    
if __name__ == '__main__':
    import torch.nn as nn
    class LinearModel(nn.Module):
        def __init__(self, h_dims):
            super(LinearModel, self).__init__()

            models = []
            for i in range(len(h_dims) - 1):
                models.append(nn.Linear(h_dims[i], h_dims[i + 1]))
                if i != len(h_dims) - 2:
                    models.append(nn.ReLU()) 
            self.models = nn.Sequential(*models)
        def forward(self, X):
            return self.models(X)
    
    test_sample_pool = [LinearModel([10, 5, 1]) for i in range(10)]
    a = ServerAggregator()
    server = serverSimulator(a, args={'node_config': {}})
    for i, model in enumerate(test_sample_pool):
        upload_param = {'client_id': i, 'state_dict': model.state_dict()}
        server.upload_model(upload_param)
    
    final_model = server.download_model()
    print("Final model structure:", final_model)