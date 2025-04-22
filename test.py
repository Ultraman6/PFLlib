import argparse


# 自定义Action类
class PopValueAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, default=None, **kwargs):
        # 初始化时接收默认值
        super().__init__(option_strings, dest, nargs, **kwargs)
        self.default = default  # 设置默认值
        self.values = []

    def __call__(self, parser, namespace, values, option_string=None):
        # 每次调用时将所有的值存储到values列表
        if not self.values:
            self.values = values  # 只第一次接收到这些值

        # 如果列表为空，使用默认值
        if not self.values:
            setattr(namespace, self.dest, self.default)
        else:
            # 获取并移除列表中的第一个值
            current_value = self.values.pop(0)  # 返回并移除首位元素
            setattr(namespace, self.dest, current_value)


def parse_args():
    parser = argparse.ArgumentParser()

    # 使用自定义的PopValueAction来控制每次返回一个值
    parser.add_argument('--dataset', type=str, nargs='+', action=PopValueAction, default='default_dataset',
                        help='List of datasets')

    # 解析参数
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # 每次访问args.dataset时，都会返回并移除首位的值
    print("First access:", args.dataset)
    print("Second access:", args.dataset)
    print("Third access:", args.dataset)
    print("Fourth access (should return default):", args.dataset)
