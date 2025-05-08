"""
VIDEO LABELER — multi‑class, resolution‑consistent

 ▣ Pause/Resume ▣ Delete‑mode ▣ Finish/q ▣
Keys 1/2/3 select class (Player/Name/Investment)
Each rectangle shows its index‑and‑class tag:  4‑I ,  2‑P , …
"""

import cv2, os

# ╔══════════════════════════● SETTINGS ●════════════════════════╗
video_path  = r'D:\Programmes\Freelance\Poker-Insight\Game\input.mp4'
output_txt  = 'boxes.txt'
max_width, max_height = 1280, 800
BTN_W, BTN_H = 90, 40
FONT = cv2.FONT_HERSHEY_SIMPLEX
# three classes:  id → (name, letter, colour BGR)
CLASSES = {
    0: ('Player',     'P', (  0,255,255)),   # Yellow
    1: ('Name',       'N', (255,  0,255)),   # Magenta
    2: ('Investment', 'I', (255,255,  0)),   # Cyan
}
# ╚═══════════════════════════════════════════════════════════════╝


# ──────────────────────────  HELPERS  ────────────────────────── #
def get_video_properties(cap):
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    return ow, oh, fps


def calculate_scale(ow, oh):
    return min(max_width/ow, max_height/oh)


def scale_coords(x, y, s):   return int(x*s), int(y*s)
def unscale_coords(x, y, s): return int(x/s), int(y/s)
def inside(x,y,x1,y1,x2,y2): return x1<=x<=x2 and y1<=y<=y2


def corner_hit(x, y, rect, tol=10):
    x1,y1,x2,y2,_ = rect
    for i,(cx,cy) in enumerate([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]):
        if abs(x-cx)<=tol and abs(y-cy)<=tol: return i
    return None
# ─────────────────────────────────────────────────────────────── #


def load_existing_boxes():
    boxes={}
    if os.path.exists(output_txt):
        with open(output_txt,'r') as f:
            for ln in f:
                parts = list(map(int,ln.strip().split(',')))
                if len(parts)==6:
                    frm,x1,y1,x2,y2,c = parts
                elif len(parts)==5:          # legacy file
                    frm,x1,y1,x2,y2 = parts
                    c=0
                else: continue
                boxes.setdefault(frm,[]).append((x1,y1,x2,y2,c))
        print(f"Loaded {sum(len(v) for v in boxes.values())} boxes")
    return boxes


# ──────────────────────────  VIEW‑ONLY  ───────────────────────── #
def run_viewer_mode(boxes):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(video_path)
    ow,oh,fps = get_video_properties(cap)
    s=calculate_scale(ow,oh);  dw,dh = int(ow*s),int(oh*s)

    all_boxes=[b for fb in boxes.values() for b in fb]

    cv2.namedWindow("Viewer",cv2.WINDOW_NORMAL); cv2.resizeWindow("Viewer",dw,dh)
    while cap.isOpened():
        ok,frame=cap.read()
        if not ok: break
        fidx=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
        disp=cv2.resize(frame,(dw,dh))

        for idx,(x1,y1,x2,y2,c) in enumerate(all_boxes,1):
            sx1,sy1=scale_coords(x1,y1,s); sx2,sy2=scale_coords(x2,y2,s)
            colour=CLASSES[c][2]
            cv2.rectangle(disp,(sx1,sy1),(sx2,sy2),colour,2)
            cv2.putText(disp,f"{idx}-{CLASSES[c][1]}",(sx1+3,sy1+18),
                        FONT,0.6,colour,2,cv2.LINE_AA)

        cv2.putText(disp,f"Frame {fidx}",(10,30),FONT,0.7,(0,255,0),2)
        cv2.imshow("Viewer",disp)
        k=cv2.waitKey(int(1000/fps))&0xFF
        if k==ord('q'): break
        if k==ord(' '):            # pause
            while True:
                k=cv2.waitKey(0)&0xFF
                if k in (ord(' '),ord('q')):
                    if k==ord('q'): cap.release(); cv2.destroyAllWindows(); return
                    break
    cap.release(); cv2.destroyAllWindows()


# ─────────────────────────  LABEL / EDIT  ─────────────────────── #
def run_labeler_mode(init=None):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(video_path)
    ow,oh,fps = get_video_properties(cap)
    s=calculate_scale(ow,oh);  dw,dh=int(ow*s),int(oh*s); delay=int(1000/(fps*2))

    # state
    boxes={f:list(rs) for f,rs in (init or {}).items()}
    cls_cur=0
    paused=finished=delete_mode=False
    drawing=moving=resizing=False
    start_pt=current_pt=move_off=None
    sel_idx=sel_corner=None

    def on_mouse(ev,x,y,flags,param):
        nonlocal drawing,moving,resizing,start_pt,current_pt
        nonlocal move_off,sel_idx,sel_corner,paused,finished,delete_mode
        ox,oy=unscale_coords(x,y,s)

        if ev==cv2.EVENT_LBUTTONDOWN:
            # GUI buttons
            if inside(x,y,10,10,10+BTN_W,10+BTN_H): paused=not paused; return
            if inside(x,y,120,10,120+BTN_W,10+BTN_H): finished=True; return
            if inside(x,y,230,10,230+BTN_W,10+BTN_H): delete_mode=not delete_mode; return
            if not paused: return

            fb=boxes.setdefault(fidx,[])
            # resize
            for i,r in enumerate(fb):
                c=corner_hit(ox,oy,r)
                if c is not None: sel_idx,sel_corner=i,c; resizing=True; return
            # move / delete
            for i,(x1,y1,x2,y2,cl) in enumerate(fb):
                if inside(ox,oy,x1,y1,x2,y2):
                    if delete_mode: del fb[i]; return
                    moving=True; sel_idx=i; move_off=(ox-x1,oy-y1); return
            # new
            drawing=True; start_pt=current_pt=(ox,oy)

        elif ev==cv2.EVENT_MOUSEMOVE:
            if drawing: current_pt=(ox,oy)
            elif resizing:
                x1,y1,x2,y2,cl = boxes[fidx][sel_idx]
                if sel_corner==0: x1,y1=ox,oy
                elif sel_corner==1: x2,y1=ox,oy
                elif sel_corner==2: x2,y2=ox,oy
                elif sel_corner==3: x1,y2=ox,oy
                x1,x2=sorted((x1,x2)); y1,y2=sorted((y1,y2))
                boxes[fidx][sel_idx]=(x1,y1,x2,y2,cl)
            elif moving:
                x1,y1,x2,y2,cl = boxes[fidx][sel_idx]
                w,h=x2-x1,y2-y1
                x1=ox-move_off[0]; y1=oy-move_off[1]
                boxes[fidx][sel_idx]=(x1,y1,x1+w,y1+h,cl)

        elif ev==cv2.EVENT_LBUTTONUP:
            if drawing:
                x1,y1=start_pt; x2,y2=current_pt
                if abs(x2-x1)>5 and abs(y2-y1)>5:
                    x1,x2=sorted((x1,x2)); y1,y2=sorted((y1,y2))
                    boxes.setdefault(fidx,[]).append((x1,y1,x2,y2,cls_cur))
            drawing=moving=resizing=False; sel_idx=sel_corner=None

    cv2.namedWindow("Labeler",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labeler",dw,dh+BTN_H+20)
    cv2.setMouseCallback("Labeler",on_mouse)

    fidx=0
    while cap.isOpened() and not finished:
        if not paused:
            ok,frame=cap.read()
            if not ok: break
            fidx=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1

        disp=cv2.resize(frame,(dw,dh))

        # buttons
        cv2.rectangle(disp,(10,10),(10+BTN_W,10+BTN_H),
                      (0,255,0) if paused else (0,0,255),-1)
        cv2.putText(disp,'Resume' if paused else 'Pause',
                    (16,10+BTN_H-10),FONT,0.55,(255,255,255),1)
        cv2.rectangle(disp,(120,10),(120+BTN_W,10+BTN_H),(255,0,0),-1)
        cv2.putText(disp,'Finish',(126,10+BTN_H-10),FONT,0.55,(255,255,255),1)
        cv2.rectangle(disp,(230,10),(230+BTN_W,10+BTN_H),
                      (0,0,255) if delete_mode else (255,0,255),-1)
        cv2.putText(disp,'Delete',(236,10+BTN_H-10),FONT,0.55,(255,255,255),1)

        # class indicator
        cname,letter,colour=CLASSES[cls_cur]
        cv2.putText(disp,f"Class: {cname} ({letter})",
                    (10,dh-10),FONT,0.6,colour,2)

        # saved boxes
        for idx,(x1,y1,x2,y2,cl) in enumerate(boxes.get(fidx,[]),1):
            sx1,sy1=scale_coords(x1,y1,s); sx2,sy2=scale_coords(x2,y2,s)
            col=CLASSES[cl][2]
            cv2.rectangle(disp,(sx1,sy1),(sx2,sy2),col,2)
            cv2.putText(disp,f"{idx}-{CLASSES[cl][1]}",
                        (sx1+3,sy1+18),FONT,0.6,col,2,cv2.LINE_AA)

        # temp box
        if drawing:
            sx1,sy1=scale_coords(*start_pt,s); sx2,sy2=scale_coords(*current_pt,s)
            cv2.rectangle(disp,(sx1,sy1),(sx2,sy2),colour,1)

        cv2.imshow("Labeler",disp)

        k=cv2.waitKey(delay if not paused else 30)&0xFF
        if k==ord('q'): finished=True
        elif k in (ord('1'),ord('2'),ord('3')):
            cls_cur=int(chr(k))-1   # map '1'→0 etc.

    cap.release(); cv2.destroyAllWindows()

    with open(output_txt,'w') as f:
        for frm,rs in boxes.items():
            for x1,y1,x2,y2,cl in rs:
                f.write(f"{frm},{x1},{y1},{x2},{y2},{cl}\n")
    print("Saved:",os.path.abspath(output_txt))


# ─────────────────────────────── MAIN ─────────────────────────── #
if __name__=="__main__":
    existing=load_existing_boxes()
    if existing:
        print("1 View  2 Edit"); ch=input("Choice: ").strip()
        if ch=="1": run_viewer_mode(existing)
        else:       run_labeler_mode(existing)
    else: run_labeler_mode()
