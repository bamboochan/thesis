import numpy as np
import torch
import torch.nn.functional as F

def feature_reproject(features, depth,rages, device):
  features_out = {}
  for feat_key in features:
    batch_size, time_size, channel, height, width = features[feat_key].shape
    H, W = height, width
    dtype = torch.float
    base_grid = torch.stack((torch.linspace(-1, 1, W, dtype=dtype).unsqueeze(0).repeat(H, 1),
                             torch.linspace(-1, 1, H, dtype=dtype).unsqueeze(-1).repeat(1, W),), dim=-1)
    base_grid = base_grid.unsqueeze(0).to(device)
    feat_out = []
    for tt in range(time_size-1):
      old_mask = features[feat_key][:, tt]
      rage2 = rages[:,tt]
      rage1 = rages[:,time_size-1]
      # reproject from t to t-1
      down_flow = get_reproject_flow(old_mask,depth, rage1, rage2,device, is_mask=False)
      #down_flow = F.interpolate(flows[:,tt].permute(0,-1,1,2), (H, W))
      #down_flow = down_flow.permute(0,2,3,1) + base_grid
      feat_t = F.grid_sample(features[feat_key][:, tt],
                             down_flow, mode='bilinear',
                             padding_mode='border',
                             align_corners=True)
      feat_out.append(feat_t.unsqueeze(1))
    feat_out.append(features[feat_key][:, -1].unsqueeze(1))
    feat_out = torch.cat(feat_out,1)
    features_out[feat_key] = feat_out
  return features_out

def get_reproject_flow(old_mask, depth, rage1, rage2,device, is_mask=False):
  # warp old_mask by reprojection; depth will be interpolated in the size of old_mask
  # return: flow-field
  b,c,h,w = old_mask.shape
  depth = F.interpolate(depth,size=[h,w])
  if is_mask:
    # if the input is mask, we only need to warp the foreground pixels
    grid = torch.nonzero(old_mask)
    py,px = grid[:,0], grid[:,1]
  else:
    py,px = torch.meshgrid(torch.tensor(range(h),device=device),torch.tensor(range(w),device=device))
    py = py.reshape((1,-1))
    px = px.reshape((1,-1))
    py = torch.cat([py]*b,0)
    px = torch.cat([px]*b,0)
  VP = torch.bmm(torch.inverse(rage1[:,0,:,:]),rage1[:,2,:,:])
  VP_inverse = torch.inverse(VP) # NDC to world coordinate

  ndcz = depth.reshape((b,-1))
  ndcx, ndcy = pixels_to_ndcs(px, py,size=(h,w))
  ndc_coord = torch.stack([ndcx,ndcy,ndcz,torch.ones_like(ndcz)], dim=2)

  # To world.
  world_coord = torch.bmm(ndc_coord , VP_inverse)
  world_coord = world_coord/world_coord[:,:,-1:]

  # Reproject to NDC
  VP2 = torch.bmm(torch.inverse(rage2[:,0,:,:]),rage2[:,2,:,:])
  final = torch.bmm(world_coord , VP2)
  final = final/final[:,:,-1:]
  pixel_y, pixel_x = ndcs_to_pixel(final[:,:,0], final[:,:,1], size=(h,w))
  yy = torch.round(pixel_y)
  xx = torch.round(pixel_x)
  yy =yy.long()
  xx =xx.long()

  yy = yy*2/h -1
  xx = xx*2/w -1
  flow = torch.stack((xx,yy),dim=2)
  flow = flow.reshape((b,h,w,2))
  return flow

def load_rage(path):
  rage_matrices = np.fromfile(path,dtype=np.float32) # read the rage matrices
  rage_matrices = rage_matrices.reshape((4,4,4))
  rage_matrices = torch.from_numpy(rage_matrices)
  return rage_matrices 

def pixels_to_ndcs(xx, yy, size=(800,1280)):
  s_y, s_x = size
  s_x -= 1  # so 1 is being mapped into (n-1)th pixel
  s_y -= 1  # so 1 is being mapped into (n-1)th pixel
  x = (2 / s_x) * xx - 1
  y = (-2 / s_y) * yy + 1
  return x, y

def ndcs_to_pixel(ndc_x, ndc_y, size=(800,1280)):
    s_y, s_x = size
    s_y -= 1  # so 1 is being mapped into (n-1)th pixel
    s_x -= 1  # so 1 is being mapped into (n-1)th pixel
    return (-(s_y / 2) * ndc_y + (s_y / 2), (s_x / 2) * ndc_x + (s_x / 2))