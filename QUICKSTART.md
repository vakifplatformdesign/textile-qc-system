# ⚡ Quick Start Guide

## 🎯 Get Running in 5 Minutes!

### **Step 1: Install Dependencies**
```bash
cd C:\Users\Pau\Desktop\SPECTROPHOTOMETER
pip install -r requirements.txt
```

### **Step 2: Run the App**
```bash
python app.py
```

### **Step 3: Open Your Browser**
```
http://localhost:5000
```

### **Step 4: Upload & Analyze**
1. Click to upload reference image
2. Click to upload test image
3. (Optional) Click "Advanced Settings" to customize
4. Click "Start Analysis"
5. Wait ~30-60 seconds
6. Download both PDF reports!

---

## 🚀 Deploy to Render (10 Minutes)

### **1. Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit - Textile QC System v1.1.0"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

### **2. Deploy on Render**
1. Go to: https://dashboard.render.com/
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: textile-qc-system
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click "Create Web Service"
6. Wait 5-10 minutes
7. **Done!** Your app is live! 🎉

---

## 📂 Important Files

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application |
| `templates/index.html` | Beautiful UI |
| `requirements.txt` | Dependencies |
| `Procfile` | Render config |
| `colabcoderefernce.py` | PDF generation (KEPT) |
| `README.md` | Full documentation |
| `DEPLOYMENT_GUIDE.md` | Detailed deployment |
| `PROJECT_SUMMARY.md` | Complete overview |

---

## ✅ Verification Checklist

Before deploying, verify:
- [ ] Dependencies install without errors
- [ ] App runs locally on port 5000
- [ ] Homepage loads correctly
- [ ] Both images can be uploaded
- [ ] Settings modal opens/closes
- [ ] Analysis completes successfully
- [ ] Both PDFs download
- [ ] PDFs match original format

---

## 🎨 What You Got

✨ **Modular Architecture** - 50+ organized files
✨ **Beautiful UI** - Modern design with brand colors
✨ **Advanced Settings** - Modal with 4 tabs
✨ **Conditional Execution** - Only runs enabled sections
✨ **Exact PDF Output** - Same as original
✨ **Deployment Ready** - GitHub + Render
✨ **Complete Docs** - Everything explained

---

## 💡 Pro Tips

1. **Copy Logo Files** to project root:
   - `llogo_square_with_name_1024x1024.png`
   - `logo_vertical_512x256.png`

2. **Disable Unused Sections** in settings to speed up analysis

3. **Use Paid Tier** on Render for faster processing

4. **Monitor Logs** to catch any issues early

---

## 🆘 Quick Troubleshooting

**App won't start?**
→ Check Python version (needs 3.11+)
→ Install dependencies: `pip install -r requirements.txt`

**Analysis fails?**
→ Check image formats (JPG, PNG, BMP, TIFF)
→ Try with smaller images first

**PDFs not downloading?**
→ Check browser pop-up blocker
→ Look in downloads folder

---

## 📚 More Information

- **Full Documentation**: See `README.md`
- **Deployment Guide**: See `DEPLOYMENT_GUIDE.md`
- **Project Overview**: See `PROJECT_SUMMARY.md`

---

# 🎉 That's It!

You're ready to use your professional textile QC system!

**Happy analyzing!** 🎨✨

