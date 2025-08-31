from fastapi import FastAPI, UploadFile, Form, HTTPException
from typing import Optional, List
from fastapi.responses import FileResponse, JSONResponse
import subprocess, uuid, os, shutil, asyncio, requests, zipfile

app = FastAPI()

WORKDIR = os.getnv("WORKDIR", "/workspace")
REPO_DIR = os.getenv("REPO_DIR", os.path.join(WORKDIR, "videocrafter"))
CHECKPOINT_T2V = os.getenv("CKPT_T2V", os.path.join(WORKDIR, "checkpoints", "base_512_v2", "model.ckpt"))
CHECKPOINT_I2V = os.getenv("CKPT_I2V", os.path.join(WORKDIR, "checkpoints", "i2v_512_v1", "model.ckpt"))
INPUTS_DIR =os.path.join(WORKDIR, "inputs")
OUTPUTS_DIR = os.path.join(WORKDIR, "outputs")
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs("videocrafter", exist_ok=True)
zip_url = "https://huggingface.co/VideoCrafter/base_512_v2/resolve/main/videocrafter.zip"
zip_path = "videocrafter/videocrafter.zip"
if not os.path.exists(zip_path):
    print(f"Downloading model repo from {zip_url}...")
    r = requests.get(zip_url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("videocrafter")
    print("Download complete.")

def try_command_variants(variants: List[List[str]], cwd: Optional[str] = None, timeout: int = 600):
    """ variants: list of command lists, tries sequentially until one succeeds (returncode == 0).
    returns tuple(success_bool, chosen_command, stdout, stderr, returncode)
    """
    for cmd in variants:
        try:
            proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            if proc.returncode == 0:
                return True, cmd, proc.stdout, proc.stderr, proc.returncode
            last = (False, cmd, proc.stdout, proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired as e:
            return False, cmd, "", f"Timed out after {timeout}s: {str(e)}", 124
        except Exception as e:
            return False, cmd, "", f"Exception while running command: {str(e)}", 1
    return last if 'last' in locals() else (False, [], "", "No command variants provided", 1)

@app.post("/generate")
async def generate_video(
    prompt: str = Form(...),
    image: UploadFile = None,
    seconds: int = Form(10),
):
    if not prompt or prompt.strip() == "":
        raise HTTPException(status_code=400, detail= "prompt is required and must be non-empty")
    req_id = str(uuid.uuid4())
    output_fileName = f"{req_id}.mp4"
    output_path = os.path.join(OUTPUTS_DIR, output_fileName)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            pass
    frames = max(1, int(seconds) * 20)
    fps = 8
    img_path = None
    if image: 
        img_ext = os.path.splitext(image.filename)[1] or ".png"
        img_path = os.path.join(INPUTS_DIR, f"{req_id}{img_ext}")
        with open(img_path, "wb") as f:
            f.write(await image.read())
    scripts_dir = os.path.join(REPO_DIR, "scripts")
    eval_txt_candidates = []
    eval_img_candidates = []
    txt_script_names = ["eval_txt2vid.py", "eval_text2video.py", "eval_t2v.py"]
    img_script_names = ["eval_img2vid.py", "eval_img2video.py", "eval_i2v.py"]
    txt_variants = []
    img_variants = []
    for s in txt_script_names:
        possible_paths = [
            os.path.join(scripts_dir, s),
            os.path.join(REPO_DIR, s),
        ]
        for pth in possible_paths:
            txt_variants.append([
                ["python", pth, "--ckpt", CHECKPOINT_T2V, "--prompt", prompt,
                                 "--output", output_path, "--W", "512", "--H", "512", "--frames",
                                 str(frames)]
                [
                   "python", pth, "--ckpt_path", CHECKPOINT_T2V, "--prompt", prompt,
                                 "--output_path", output_path, "--W", "512", "--H", "512", "--max_frames",
                                 str(frames)
                ]
                [
                   "python", pth, "--ckpt_path", CHECKPOINT_T2V, "--prompt", prompt,
                                 "--out", output_path, "--frames", str(frames)
                 ]
                ])
            for s in img_script_names:
                possible_paths = [
                    os.path.join(scripts_dir, s),
                    os.path.join(REPO_DIR, s),
                ]
                for pth in possible_paths:
                    img_variants.append([
                        ["python", pth, "--ckpt", CHECKPOINT_I2V,  "--input_image", img_path, "--prompt", prompt,
                                         "--output", output_path, "--W", "512", "--H", "512", "--frames",
                                         str(frames)]
                        [
                           "python", pth, "--ckpt_path", CHECKPOINT_I2V,
                           "--input_image", img_path, "--prompt", prompt,
                                     "--output_path", output_path, "--max_frames", str(frames)
                        ]
                        [
                           "python", pth, "--ckpt_path", CHECKPOINT_I2V,  "--input_image", img_path, "--prompt", prompt,
                                         "--out", output_path, "--frames", str(frames)
                         ]
                        ])
            flat_variants = [] 
            if img_path:
                for group in img_variants:
                   for cmd in group:
                          flat_variants.append(cmd)
                for group in txt_variants:
                   for cmd in group:
                          flat_variants.append(cmd)
            else: 
                for group in txt_variants:
                    for cmd in group:
                        flat_variants.append(cmd)
            if not flat_variants:
                raise HTTPException(status_code=500, detail="No valid command variants found for execution.")
            success, used_cmd, stdout, stderr, rc, = try_command_variants(flat_variants, cwd=REPO_DIR, timeout=600)
            if not success:
                detail = {
                    "error": "All command variants failed",
                    "last_cmd_tried" : used_cmd,
                    "stdout": stdout,
                    "stderr": stderr,
                    "checkpoints" : {
                        "t2v_exists": os.path.exists(CHECKPOINT_T2V),
                        "i2v_exists": os.path.exists(CHECKPOINT_I2V)
                    },
                    "repo_dir": REPO_DIR,
                }
                if img_path and os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except: pass
                    return JSONResponse(status_code=500, content=detail)
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 20:
                candidates = sorted([os.path.join(OUTPUTS_DIR, f) for f in os.listdir(OUTPUTS_DIR) if f.endswith(".mp4")], key=os.path.getmtime, reverse=True)
                if candidates:
                    output_path = candidates[0]
                else:
                    return JSONResponse(status_code=500, content={
                        "error": "Output video not found after successful command execution.",
                        "used_cmd": used_cmd,
                        "stdout": stdout,
                        "stderr": stderr,
                        })
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except: pass
            return FileResponse(output_path, media_type="video/mp4", filename= os.path.basename(output_path))
       
@app.get("/health")
def health():
    return {"status": "ok", "checkpoints_exist": {
        "t2v": os.path.exists(CHECKPOINT_T2V),
        "i2v": os.path.exists(CHECKPOINT_I2V)
    }}