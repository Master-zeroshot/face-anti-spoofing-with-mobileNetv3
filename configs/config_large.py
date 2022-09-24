exp_num = 0

dataset = 'celeba_spoof'

multi_task_learning = True

evaluation = True

test_steps = None

random_seed = 42

datasets = dict(Celeba_root='../CelebA_Spoof',
                Casia_root='./CASIA',
                LCCFASD_root='../LCC_FASD')

img_norm_cfg = dict(mean=[0.5931, 0.4690, 0.4229],
                    std=[0.2471, 0.2214, 0.2157])

optimizer = dict(lr=0.005, momentum=0.9, weight_decay=5e-4)

scheduler = dict(milestones=[20,50], gamma=0.2)

data = dict(batch_size=150,
            data_loader_workers=32,
            sampler=True,
            pin_memory=True)

resize = dict(height=224, width=224)

checkpoint = dict(snapshot_name="MobileNet3-large-224.pth.tar",
                  experiment_path='./logs_large')

loss = dict(loss_type='amsoftmax',
            amsoftmax=dict(m=0.5,
                           s=1,
                           margin_type='cross_entropy',
                           label_smooth=False,
                           smoothing=0.1,
                           ratio=[1,1],
                           gamma=0)
            )

epochs = dict(start_epoch=0, max_epoch=71)

model= dict(model_size = 'large',
            width_mult = 1.0,
            pretrained=True,
            embeding_dim=1280,
            imagenet_weights='./pretrained/mobilenetv3-large-1cd25616.pth')

aug = dict(type_aug=None,
            alpha=0.5,
            beta=0.5,
            aug_prob=0.7)

curves = dict(det_curve='det_curve_large.png',
              roc_curve='roc_curve_large.png')

dropout = dict(prob_dropout=0.2,
               classifier=0.3,
               type='gaussian',
               mu=0.5,
               sigma=0.3)

RSC = dict(use_rsc=False,
           p=0.333,
           b=0.333)

test_dataset = dict(type='LCC_FASD')

conv_cd = dict(theta=0.7)

test_file_name = "test_mnv3_large"
