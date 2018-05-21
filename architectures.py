from DensenetModels import DenseNet121, DenseNet169,  DenseNet201, ResNet18


def load_model(model_name, pretrained = True):
    architectures = {
    'DENSE-NET-121': DenseNet121(nnClassCount,pretrained).cuda(),
    'DENSE-NET-169': DenseNet169(nnClasscount, pretrained).cuda(),
    'DENSE-NET-201': DenseNet201(nnClassCount, pretrained).cuda(),
    'RESNET-18': ResNet18(nnClassCount, pretrained).cuda()
    }
    return architectures[model_name]
