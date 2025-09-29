import timm
print([m for m in timm.list_models() if "mobile" in m or "shufflenet" in m or "efficient" in m])

