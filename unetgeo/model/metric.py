import torch
import torchmetrics

import unetgeo.utils.analysis_utils as au

class UnetGeoMetric(torchmetrics.Metric):
    def __init__(self, n_relation_type, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # self.add_state("f1", default=torch.FloatTensor(0), dist_reduce_fx="mean")  # parse
        self.add_state(
            "tp_edge", default=torch.zeros(n_relation_type), dist_reduce_fx="sum"
        )
        self.add_state(
            "fp_edge", default=torch.zeros(n_relation_type), dist_reduce_fx="sum"
        )
        self.add_state(
            "fn_edge", default=torch.zeros(n_relation_type), dist_reduce_fx="sum"
        )
        self.add_state("tp_parse", default=torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("fp_parse", default=torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("fn_parse", default=torch.zeros(()), dist_reduce_fx="sum")

        # this is used as accuracy of segmentation instead of f1_edge
        self.add_state("seg_acc", default=torch.zeros(()), dist_reduce_fx="sum")
        # Counter will increase per example 
        self.add_state("counter", default=torch.zeros(()), dist_reduce_fx="sum")

    def update(self, tp_edge, fp_edge, fn_edge, tp_parse, fp_parse, fn_parse, seg_acc):
        self.tp_edge += torch.tensor(tp_edge).type_as(self.tp_edge)
        self.fp_edge += torch.tensor(fp_edge).type_as(self.fp_edge)
        self.fn_edge += torch.tensor(fn_edge).type_as(self.fn_edge)
        self.tp_parse += torch.tensor(tp_parse).type_as(self.tp_parse)
        self.fp_parse += torch.tensor(fp_parse).type_as(self.fp_parse)
        self.fn_parse += torch.tensor(fn_parse).type_as(self.fn_parse)

        self.seg_acc += torch.tensor(seg_acc).type_as(self.seg_acc)
        self.counter += torch.tensor(1).type_as(self.counter)

    def compute(self):
        self.tp_edge = self.tp_edge / self.counter
        self.fp_edge = self.fp_edge / self.counter
        self.fn_edge = self.fn_edge / self.counter 
        self.tp_parse = self.tp_parse / self.counter
        self.fp_parse = self.fp_parse / self.counter
        self.fn_parse = self.fn_parse / self.counter

        p_parse, r_parse, f1_parse = au.my_cal_p_r_f1(
            self.tp_parse,self.fn_parse, self.fp_parse
        )
        p_edge_avg, r_edge_avg, f1_edge_avg = au.my_cal_p_r_f1(
            self.tp_edge, self.fn_edge, self.fp_edge
        )

        acc = self.seg_acc / self.counter

        return p_edge_avg, r_edge_avg, f1_edge_avg, p_parse, r_parse, f1_parse, acc
