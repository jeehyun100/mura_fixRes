import timm
from pprint import pprint
model_names = timm.list_models('*effi*b*')
pprint(model_names)