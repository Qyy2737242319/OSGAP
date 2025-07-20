import yaml

class EasyConfig:
    def __init__(self, config_file):
        """
        从 YAML 配置文件中读取内容并实例化 EasyConfig 对象。

        参数:
        - config_file (str): 配置文件路径。
        """
        with open(config_file, 'r') as file:
            self.config_dict = yaml.safe_load(file)

    def __getattr__(self, name):
        """
        允许通过属性访问配置值。

        参数:
        - name (str): 配置的键名。

        返回:
        - 对应配置的值，如果键不存在则引发 AttributeError。
        """
        if name in self.config_dict:
            return self.config_dict[name]
        raise AttributeError(f"'EasyConfig' object has no attribute '{name}'")
    
    def __repr__(self):
        return f"EasyConfig({self.config_dict})"