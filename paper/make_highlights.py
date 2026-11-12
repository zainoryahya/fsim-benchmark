"""Write highlights.docx (KBS: 3-5 bullet points, at most 85 characters each)."""
import docx
H = ["Filters or SMOTE placed before cross-validation inflate null MCC by up to 0.55",
     "A moved decision threshold gives the G-mean gain usually credited to SMOTE",
     "No filter or embedded method was significantly better than a Hellinger filter",
     "Grey wolf and genetic wrappers beat the filter only with their fitness classifier",
     "Wrapper subsets were unstable; more search or multitasking gave no gain"]
assert all(len(h) <= 85 for h in H), [len(h) for h in H]
d = docx.Document()
d.add_heading("Highlights", level=1)
for h in H:
    d.add_paragraph(h, style="List Bullet")
d.save("highlights.docx")
print([len(h) for h in H])
