import ruamel.yaml
yaml = ruamel.yaml.YAML()  

from easydict import EasyDict as edict
config = edict(yaml.load(open('config/config.yaml', 'r', encoding='utf-8')))
print(config.myname) # 输出 my_name
print(config) # 输出 my_name
def experiment():
    pass 

if __name__ == '__main__':
    experiment()
    