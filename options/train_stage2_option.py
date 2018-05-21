from .base_options import BaseStage2Options


class TrainStage2Options(BaseStage2Options):
    def initialize(self):
        BaseStage2Options.initialize(self)
        # pre-trained model path
        self.parser.add_argument('--G_path', type=str, default='./pretrained_models/latest_net_G.pth', help='which generator G to load')
        self.parser.add_argument('--E_path', type=str, default='./pretrained_models/latest_net_E.pth', help='which encoder E to load')
        # base train option
        self.parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        self.parser.add_argument('--update_html_freq', type=int, default=4000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=50,help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true',help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lambda_weight_panelty', type=float, default=10, help='weight of gradient panelty')
        self.parser.add_argument('--which_optimizer', type=str, default='RMSprop',help='Types of optimizer:Adam|RMSprop ')
        self.parser.add_argument('--weather_random', type=bool,default=False,help='whether ignore random latent vector generating image')
        # learning rate
        self.parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam:2e-4 | RMSprop:1e-4')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--disc_iters', type=int, default=1, help='number of D updates per G update')
        # local loss lambda parameters
        self.parser.add_argument('--lambda_z', type=float, default=0, help='weight for ||E(G(random_z)) - random_z||')
        self.parser.add_argument('--lambda_kl', type=float, default=0, help='weight for KL loss')
        self.parser.add_argument('--lambda_s_l', type=float, default=1e7, help='weight for local style loss')
        self.parser.add_argument('--lambda_p_l', type=float, default=1e3, help='weight for local pixel loss')
        self.parser.add_argument('--lambda_GAN_D',type=float,default=1e6, help='weight for loss_D')
        self.parser.add_argument('--lambda_GAN_l', type=float, default=1e6, help='weight on local D loss')
        self.parser.add_argument('--lambda_GAN2_l', type=float, default=0, help='weight on local D loss')
        self.parser.add_argument('--lambda_g_l', type=float, default=1e10, help='weight for local glcm loss')  # not used
        self.parser.add_argument('--lambda_c', type=float, default=20, help='weight for content loss')
        self.isTrain = True
        # local random block
        self.parser.add_argument('--block_num', type=int, default=5, help='num of random blocks')
        self.parser.add_argument('--min_block_size', type=int, default=45, help='min size of random block')
        self.parser.add_argument('--max_block_size', type=int, default=64, help='max size of random block')
        self.isTrain = True