from argparse import Namespace
import torch.nn as nn

def build_shape_model(args: Namespace) -> nn.Module:
    if args.shape_model == 'Wu':
        from .fold.shape_layers import Wu
        return Wu(xi=0.774, mu=0.078, sigma=0.083, alpha=1.006, beta=1.404)
        
    elif args.shape_model == 'Foo':
        from .fold.shape_layers import Foo
        return Foo(p_alpha=0.540, p_beta=1.390, u_alpha=1.006, u_beta=1.404)

    elif args.shape_model == 'External':
        from .loss.external_shape_predictor import ExternalShapePredictor
        # 例: args から外部モデルロード関数やパスを受け取る実装に置き換えてください
        return ExternalShapePredictor()
    elif args.shape_model == 'ShapeTransformer':
        from .loss.predict_shape import ShapeTransformer
        return ShapeTransformer()

    elif args.shape_model == 'ShapeConv':
        from .loss.predict_shape import ShapeConv
        return ShapeConv()
        
    elif args.shape_model == 'RiboEM':
        from .fold.shape_layers import RiboEM
        return RiboEM(mu_u=0.36248604585583205,
                            sig_u=0.3004844655699528,
                            mu_p=0.0,
                            sig_p=0.10)
    elif args.shape_model == 'ContraSE':
        from .fold.shape_layers import ContraSE
        # デフォルトコンストラクタを使う。必要なら args でパラメータを渡す実装に拡張してください。
        return ContraSE()

    elif args.shape_model == 'Helix':
        from .fold.shape_layers import Helix
        # デフォルト初期値の Helix を返す
        return Helix()
    elif args.shape_model == 'Corr':
        from .fold.shape_layers import Corr
        return Corr()
    else:
        raise ValueError(f'not implemented: {args.shape_model}')
