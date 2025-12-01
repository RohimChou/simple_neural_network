try plotting Neural Network `dosage prediction` visualization in 3D space.


- Sample training data: (dosage in mg, effectiveness)  
  inputs: dosage (scalar, between 0 and 10 mg)  
  outputs: effectiveness (between 0 and 1)  
```
test_datas = [
    TestData(0, 0),
    TestData(1, 0.2),
    TestData(3, 0.9),
    TestData(5, 1),
    TestData(7, 0.7),
    TestData(10, 0.1),
]
```