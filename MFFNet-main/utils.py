def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*decay
        print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, init_lr*decay))

# def make_dir():
#     now = datetime.now()
#     #dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
#     dt_string = now.strftime("%Y-%m-%d-%H-%M")
#     save_dir = "%s_" % (dt_string)
#     return save_dir