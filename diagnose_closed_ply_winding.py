#!/usr/bin/env python3
"""Read-only diagnostic for contradictory face orientation constraints in Blender PLY files.

Run: python diagnose_closed_ply_winding.py path/to/003_back_r_leg_closed.ply
Supports the XYZ float32 / uchar uint polygon-face binary little-endian PLY
layout emitted by the previous Oktopus repair scripts. Does not edit files.
"""
import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
import re
import numpy as np


def load_binary_ply(path):
    content = path.read_bytes()
    end = content.find(b'end_header\n')
    if end < 0: raise ValueError('No end_header\\n in PLY')
    h_end = end + len(b'end_header\n')
    h = content[:h_end].decode('ascii')
    if 'format binary_little_endian 1.0' not in h:
        raise ValueError('Expected binary_little_endian 1.0 PLY')
    nv = int(re.search(r'(?m)^element vertex (\d+)$', h)[1])
    nf = int(re.search(r'(?m)^element face (\d+)$', h)[1])
    v_section = h.split('element vertex ',1)[1].split('element face ',1)[0].splitlines()[1:]
    f_section = h.split('element face ',1)[1].split('end_header',1)[0].splitlines()[1:]
    if v_section != ['property float x','property float y','property float z'] or f_section != ['property list uchar uint vertex_indices']:
        raise ValueError(f'Unexpected PLY vertex/face schema: {v_section}, {f_section}')
    xyz = np.frombuffer(content, dtype='<f4', count=nv*3, offset=h_end).reshape(-1,3).copy()
    p = h_end + nv*12
    original_faces=[]
    for i in range(nf):
        if p>=len(content): raise ValueError(f'Truncated before face {i}')
        count=content[p]
        if count<3 or p+1+4*count>len(content):raise ValueError(f'Invalid face {i} length {count}')
        face=np.frombuffer(content, dtype='<u4',count=count,offset=p+1).astype(np.int64).tolist()
        original_faces.append(face)
        p+=1+4*count
    if p!=len(content):raise ValueError(f'Unexpected trailing data: {len(content)-p} bytes')
    return xyz, original_faces


def diagnose(path):
    xyz, allfaces=load_binary_ply(path)
    tri_indices=[i for i,f in enumerate(allfaces) if len(f)==3]
    tri=[allfaces[i] for i in tri_indices]
    print('FILE:',path)
    print('VERTICES:',len(xyz),'FACES:',len(allfaces),'FACE SIZES:',dict(Counter(map(len,allfaces))))
    degenerate=[tri_indices[i] for i,t in enumerate(tri) if len(set(t))<3]
    print('DEGENERATE TRIANGLES:',len(degenerate), 'first face indices:',degenerate[:20])
    same=defaultdict(list)
    for j,t in enumerate(tri):same[tuple(sorted(t))].append(tri_indices[j])
    duplicates=[ids for ids in same.values() if len(ids)>1]
    print('DUPLICATE TRIANGLE GROUPS:',len(duplicates),'first:',duplicates[:8])
    edge_users=defaultdict(list)
    for i,t in enumerate(tri):
        for u,v in zip(t,t[1:]+t[:1]):
            edge_users[tuple(sorted((u,v)))].append((i,u,v))
    print('EDGE INCIDENCES:',dict(Counter(map(len,edge_users.values()))))
    overused=[(e,users) for e,users in edge_users.items() if len(users)>2]
    if overused:
        print('EDGES WITH >2 ORIGINAL TRIANGLES:',len(overused))
        for e,users in overused[:10]:
            print(' edge',e,'faces',[tri_indices[f] for f,_,_ in users], 'xyz',[xyz[v].tolist() for v in e])
        print('These edges require topology repair before winding analysis.')
        return
    adj=defaultdict(list)
    for e,users in edge_users.items():
        if len(users)==2:
            (a,u,v),(b,x,y)=users
            parity=int(u==x and v==y)
            adj[a].append((b,parity,e))
            adj[b].append((a,parity,e))
    state={}; parent={}; components=0; flips=0
    for seed in range(len(tri)):
        if seed in state:continue
        components+=1
        state[seed]=0;parent[seed]=None
        q=deque([seed])
        while q:
            a=q.popleft()
            for b,parity,e in adj[a]:
                required=state[a]^parity
                if b not in state:
                    state[b]=required;parent[b]=(a,e,parity)
                    q.append(b)
                elif state[b]!=required:
                    print('ORIENTATION CONTRADICTION:')
                    print(' Original PLY faces:',tri_indices[a],tri_indices[b])
                    print(' Shared edge:',e,'xyz:',[xyz[v].tolist() for v in e])
                    print(' Both faces use edge in', 'SAME' if parity else 'OPPOSITE', 'direction')
                    print(' Current parity labels:',state[a],state[b],'; required label for latter:',required)
                    ancs={}; cur=a
                    while cur is not None:
                        ancs[cur]=len(ancs)
                        cur=parent[cur][0] if parent[cur] is not None else None
                    trail=[]; cur=b
                    while cur not in ancs:
                        trail.append(cur)
                        cur=parent[cur][0]
                    lca=cur
                    chain=[];cur=a
                    while cur!=lca:
                        chain.append(cur);cur=parent[cur][0]
                    chain.append(lca)
                    chain+=trail[::-1]
                    print(' Contradictory adjacency cycle:',len(chain),'faces')
                    for k,node in enumerate(chain):
                        nxt=chain[(k+1)%len(chain)]
                        item=next((x for x in adj[node] if x[0]==nxt),None)
                        if item is None:
                            print('  Internal diagnostic error: edge absent',node,nxt);break
                        print('   PLY face %d -- edge %s (parity %d) --> PLY face %d' %
                              (tri_indices[node],item[2],item[1],tri_indices[nxt]))
                        if k>=32 and len(chain)>34:
                            print('   ... (%d further connections)'%(len(chain)-k-1));break
                    print('DO NOT use face flips alone. Inspect this region or upload the closed PLY for targeted repair.')
                    return
    flips=sum(state.values())
    print('ORIENTATION CONSTRAINTS: CONSISTENT (components=%d; flips with arbitrary component roots=%d)'%(components,flips))
    print('If v3 still failed, check that exactly the same file was tested.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ply',type=Path,nargs='+')
    args=ap.parse_args()
    for path in args.ply:
        try:diagnose(path)
        except Exception as exc:print('FAILED TO DIAGNOSE',path,type(exc).__name__,str(exc))
        print()

if __name__=='__main__': main()
