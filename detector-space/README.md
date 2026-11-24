---
title: DocuVerify Tampering Model
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# DocuVerify model service

This Space loads the MIT-licensed zodumair/document-forgery-detector vision model and exposes /predict for the upload page. The model card describes a document-level Real/Forged classifier trained with synthetic edits and ELA preprocessing. It does not localize altered pixels, and its published metrics are the model author's claims, not independently validated results for DocuVerify receipts.
