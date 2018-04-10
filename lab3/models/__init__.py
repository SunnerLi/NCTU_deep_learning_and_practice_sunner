from .ShowAttendTellModel import ShowAttendTellModel
from .WhereToLookModel import WhereToLookModel
from .ShowTellModel import ShowTellModel
from .TopDownModel import TopDownModel

def setup(opt):
    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'show_attend_tell':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'top_down':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'where_to_look':
        model = ShowTellModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))
    return model