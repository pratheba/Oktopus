import trimesh
import numpy as np
import sys
import os

def export(npzfile, fname, outfolder):
    t = np.load(npzfile, allow_pickle=True)
    allsegments = t['segments']

    outfolder = os.path.join(outfolder, fname)
    os.makedirs(outfolder,exist_ok=True)


    for i, seg in enumerate(allsegments):
        #print(i)
        for k, v in seg.items():
            #try:
                if k in ["keypoints", "surface_points_all", "surface_points_owned"]:
                    #print("yes"i)
                    outfile = os.path.join(outfolder, str(i)+'_'+str(k)+'.ply')
                    trimesh.Trimesh(vertices = v, process=False).export(outfile)

            #except:
            #    print(k)
            


if __name__ == '__main__':
    export(*sys.argv[1:])
