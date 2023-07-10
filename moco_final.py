from moco_scripts import gen_template,for_field_solver,inv_field_solver,inv_field_solver_refine,gen_MSLR_for,gen_MSLR_adj,gen,flows,save_images_inv,f,save_images_for,rescale_flow,_updateb,warp1,MoCo_MSLR



def moco_run(ksp,coord,dcf,spokes_per_bin,T):
    
    import torch
    import random
    T=500
    spokes_per_bin=ksp.shape[1]//500
    res_scale=0.5
    nf_arr = np.sqrt(np.sum(coord[0][0, :, :] ** 2, axis=1))
    RO = np.sum(nf_arr < np.max(nf_arr) * res_scale)
    im_testa0,mps0,kspa0,coorda0,dcfa0=gen_template(ksp,coord,dcf,RO,spokes_per_bin)

    res_scale=1
    nf_arr = np.sqrt(np.sum(coord[0][0, :, :] ** 2, axis=1))
    RO = np.sum(nf_arr < np.max(nf_arr) * res_scale)
    im_testaF,mpsF,kspaF,coordaF,dcfaF=gen_template(ksp,coord,dcf,RO,spokes_per_bin)
    
   
    rank=1
    scale=1
    block_size_adj=[150,170,220]
    block_size_for=block_size_adj
    T=500
    rank=1
    scale=1
    deformL_param_for0,deformR_param_for0,block_torch,ishape=gen_MSLR_for(T,rank,block_size_adj,block_size_for,scale,mps0)

    T=500
    rank=1
    scale=1
    block_size_adj=[150,170,220]
    block_size_for=block_size_adj
    T=500
    rank=1
    scale=1
    deformL_param_adj0,deformR_param_adj0,block_torch,ishape=gen_MSLR_adj(T,rank,block_size_adj,block_size_for,scale,mps0)

    T=500
    rank=1
    scale=1
    block_size_adj=[150,170,220]
    block_size_for=block_size_adj
    T=500
    rank=1
    scale=1
    deformL_param_adj1,deformR_param_adj1,block_torch,ishape=gen_MSLR_adj(T,rank,block_size_adj,block_size_for,scale,mps0)
    
    import torch
    import random  
    slice0=50
    template=im_testa0
    old_res=mps0
    RO=200
    iterations=40
    weight_dc=20
    weight_smoother=500
    T1=0
    T2=50
    for_field_solver(slice0,deformL_param_for0, deformR_param_for0, deformL_param_adj0, deformR_param_adj0, template, kspa0, coorda0, dcfa0, mps0, iterations, RO, block_torch, ishape, T1, T2, spokes_per_bin, weight_dc, weight_smoother,old_res)


    import sigpy as sp
    import torch
    import random
    slice0=50
    template=im_testa0
    old_res=mps0
    RO=200
    iterations=200
    weight_dc=3000
    weight_smoother=100
    T1=0
    T2=50
    deformL_param_for=[]
    deformR_param_for=[]
    for i in range(3):
        deformL_param_for.append(torch.from_numpy(deformL_param_for0[i].detach().cpu().numpy()).cuda())
        deformR_param_for.append(torch.from_numpy(deformR_param_for0[i].detach().cpu().numpy()).cuda())
    inv_field_solver(slice0,deformL_param_for, deformR_param_for, deformL_param_adj0, deformR_param_adj0, template, kspa0, coorda0, dcfa0, mps0, iterations, RO, block_torch, ishape, T1, T2, spokes_per_bin, weight_dc, weight_smoother,old_res)


    slice0=50
    template=im_testa0
    old_res=mps0
    RO=200
    iterations=100
    weight_dc=1e5
    weight_smoother=1
    T1=0
    T2=50
    deformL_param_for=[]
    deformR_param_for=[]
    deformL_param_adj=[]
    deformR_param_adj=[]
    old_res=mps0
    for i in range(3):
        deformL_param_for.append(torch.from_numpy(deformL_param_for0[i].detach().cpu().numpy()).cuda())
        deformR_param_for.append(torch.from_numpy(deformR_param_for0[i].detach().cpu().numpy()).cuda())
        deformL_param_adj.append(torch.from_numpy(deformL_param_adj0[i].detach().cpu().numpy()).cuda())
        deformR_param_adj.append(torch.from_numpy(deformR_param_adj0[i].detach().cpu().numpy()).cuda())
    inv_field_solver_refine(slice0,deformL_param_for, deformR_param_for, deformL_param_adj, deformR_param_adj, deformL_param_adj1, deformR_param_adj1, template, kspa0, coorda0, dcfa0, mpsF, iterations, RO, block_torch, ishape, T1, T2, spokes_per_bin, weight_dc, weight_smoother,old_res)

    low_res=mps0
    T=50
    lamda=1e-8
    end=50*kspa0.shape[1]//500
    L,R,B=MoCo_MSLR(kspa0[:,:end], coorda0[:end], dcfa0[:end], mps0, low_res,T, lamda, ishape,block_torch,deformL_param_for0 , deformR_param_for0,deformL_param_adj0,deformR_param_adj0,deformL_param_adj1,deformR_param_adj1,
                     blk_widths=[64, 128], alpha=1, beta=.5,sgw=None,
                     device=0, comm=None, seed=0,
                     max_epoch=60, decay_epoch=20, max_power_iter=1,
                     show_pbar=True).run()
    return L,R,B,deformL_param_for0,deformR_param_for0,deformL_param_adj0,deformR_param_adj0,deformL_param_adj1,deformR_param_adj1

