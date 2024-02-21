import numpy as np
from ODYM.odym.modules.dynamic_stock_model import DynamicStockModel


class MultiDim_DynamicStockModel(DynamicStockModel):
    def compute_all_stock_driven(self):
        self.compute_stock_driven_model()
        self.compute_outflow_total()
        self.compute_stock_change()
        self.check_steel_stock_dsm()

    def compute_all_inflow_driven(self):
        self.compute_s_c_inflow_driven()
        self.compute_o_c_from_s_c()
        self.compute_stock_total()
        self.compute_outflow_total()
        self.check_steel_stock_dsm()

    def check_steel_stock_dsm(self):
        balance = self.check_stock_balance()
        balance = np.abs(balance).sum()
        if balance > 1:  # 1 tonne accuracy
            raise RuntimeError("Stock balance for dynamic stock base_model is too high: " + str(balance))
        elif balance > 0.001:
            print("Stock balance for base_model dynamic stock base_model is noteworthy: " + str(balance))


def _test():
    return


if __name__ == '__main__':
    _test()
