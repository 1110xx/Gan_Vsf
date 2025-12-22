True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt = torch.load(args.pretrain_ckpt, map_location=device)
/home/yxtang/anaconda3/envs/Gan_VSF_py39/lib/python3.9/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
Traceback (most recent call last):
  File "/home/yxtang/Gan_VSF/trian_perd.py", line 451, in <module>
    main()
  File "/home/yxtang/Gan_VSF/trian_perd.py", line 447, in main
    train_loop(encoder, pred_head, decoder, train_loader, val_loader, args, scaler)
  File "/home/yxtang/Gan_VSF/trian_perd.py", line 256, in train_loop
    train_metrics = train_epoch(
  File "/home/yxtang/Gan_VSF/trian_perd.py", line 66, in train_epoch
    s_mean =scaler.mean if hasattr(scaler, 'mean') else scaler['mean']
KeyError: 'mean'