import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from system.flcore.clients.clientnh import clientNH
from system.flcore.servers.serverbase import Server
from threading import Thread
from collections import defaultdict


class FedNH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientNH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.server_model_state_dict = copy.deepcopy(self.global_model.state_dict())

        self.FedNH_server_lr_decay = args.FedNH_server_lr_decay
        self.FedNH_server_adv_prototype_agg = args.FedNH_server_adv_prototype_agg
        self.FedNH_smoothing = args.FedNH_smoothing


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models_protos()
            self.aggregate_models_protos(i)
            self.global_model.load_state_dict(self.server_model_state_dict)
            self.send_models()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models_protos(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_models = []
        self.uploaded_samples = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            self.uploaded_models.append(client.model)
            self.uploaded_samples.append(torch.tensor(list(client.train_samples_by_class.values())).to(self.device))

    def aggregate_models_protos(self, global_round: int):
        server_lr = self.learning_rate * (self.FedNH_server_lr_decay ** (global_round - 1))
        num_participants = len(self.selected_clients)
        update_direction_state_dict = None
        # agg weights for prototype
        cumsum_per_class = torch.zeros(self.num_classes).to(self.device)
        agg_weights_vec_dict = {}
        with torch.no_grad():
            for idx, (client_model, client_proto, client_samples) in enumerate(zip(self.uploaded_models, self.uploaded_protos, self.uploaded_samples)):
                if not self.FedNH_server_adv_prototype_agg:
                    cumsum_per_class += client_samples
                else:
                    mu = client_proto
                    W = list(self.server_model_state_dict.values())[-2]
                    agg_weights_vec_dict[idx] = torch.exp(torch.sum(W * mu, dim=1, keepdim=True))
                client_update = linear_combination_state_dict(client_model.state_dict(),
                                                              self.server_model_state_dict,
                                                              1.0,
                                                              -1.0)
                if idx == 0:
                    update_direction_state_dict = client_update
                else:
                    update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
                                                                                client_update,
                                                                                1.0,
                                                                                1.0)
            # new feature extractor
            self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
                                                                         update_direction_state_dict,
                                                                         1.0,
                                                                         server_lr / num_participants)
            new_head_param = list(self.server_model_state_dict.values())[-2]
            k = list(self.server_model_state_dict.keys())[-2]
            avg_prototype = torch.zeros_like(new_head_param)
            if not self.FedNH_server_adv_prototype_agg: # 此处直接上传原型参数
                for prototype in self.uploaded_protos:
                    avg_prototype += prototype / cumsum_per_class.view(-1, 1)
            else:
                m = new_head_param.shape[0]
                sum_of_weights = torch.zeros((m, 1)).to(avg_prototype.device)
                for idx, prototype in enumerate(self.uploaded_protos):
                    sum_of_weights += agg_weights_vec_dict[idx]
                    avg_prototype += agg_weights_vec_dict[idx] * prototype
                avg_prototype /= sum_of_weights

            # normalize prototype
            avg_prototype = F.normalize(avg_prototype, dim=1)
            weight = self.FedNH_smoothing
            temp = weight * new_head_param + (1 - weight) * avg_prototype

            self.server_model_state_dict[k].copy_(F.normalize(temp, dim=1))

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))

def proto_aggregation(local_protos_list):
    agg_protos = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos[label].append(local_protos[label])

    for [label, proto_list] in agg_protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos[label] = proto / len(proto_list)
        else:
            agg_protos[label] = proto_list[0].data

    return agg_protos

def linear_combination_state_dict(this, other, this_weight=1.0, other_weight=1.0, exclude=set()):
    """
        this, other: state_dict
        this_weight * this + other_weight * other
    """
    with torch.no_grad():
        ans = copy.deepcopy(this)
        for state_key in this.keys():
            if state_key not in exclude:
                ans[state_key] = this[state_key] * this_weight + other[state_key] * other_weight
        return ans