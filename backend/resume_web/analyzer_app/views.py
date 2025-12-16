import os
import sys
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Add project root to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(BASE_DIR)

from engine import run_resume_analyzer


def index(request):
    context = {}

    if request.method == "POST" and request.FILES.get("resume"):
        resume_file = request.FILES["resume"]
        jd_file = request.FILES.get("jd")

        fs = FileSystemStorage(location=os.path.join(BASE_DIR, "uploads"))
        resume_path = fs.save(resume_file.name, resume_file)
        resume_full_path = fs.path(resume_path)

        jd_full_path = None
        if jd_file:
            jd_path = fs.save(jd_file.name, jd_file)
            jd_full_path = fs.path(jd_path)

        result = run_resume_analyzer(
            resume_path=resume_full_path,
            jd_path=jd_full_path,
            verbose=False
        )

        context["result"] = result

    return render(request, "index.html", context)
