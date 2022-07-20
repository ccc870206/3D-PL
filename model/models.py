
def create_model(opt):
    print(opt.model)
    if opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()  
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model
