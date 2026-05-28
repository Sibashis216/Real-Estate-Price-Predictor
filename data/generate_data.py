# python -c "
import pandas as pd, numpy as np
np.random.seed(42)
locations=['Whitefield','Koramangala','Indiranagar','HSR Layout','Electronic City','Sarjapur Road','Marathahalli','Bannerghatta Road','JP Nagar','Jayanagar','Rajajinagar','Yeshwanthpur','Hebbal','Bellandur','BTM Layout','Yelahanka','Hennur','Horamavu','Kengeri','Uttarahalli','Vijayanagar','Basaveshwara Nagar','Malleswaram','Sadashivanagar','Brookefield','Devanahalli','Kanakapura Road','Tumkur Road','Mysore Road','Hosur Road']
lm={'Sadashivanagar':1.9,'Malleswaram':1.7,'Indiranagar':1.8,'Koramangala':1.75,'Jayanagar':1.6,'Rajajinagar':1.5,'Basaveshwara Nagar':1.5,'HSR Layout':1.55,'JP Nagar':1.4,'BTM Layout':1.35,'Yeshwanthpur':1.3,'Hebbal':1.35,'Whitefield':1.3,'Marathahalli':1.25,'Bellandur':1.3,'Sarjapur Road':1.2,'Brookefield':1.25,'Electronic City':1.1,'Bannerghatta Road':1.2,'Vijayanagar':1.25,'Yelahanka':1.1,'Hennur':1.05,'Horamavu':1.0,'Kanakapura Road':1.0,'Devanahalli':0.95,'Mysore Road':0.9,'Kengeri':0.9,'Uttarahalli':0.88,'Tumkur Road':0.85,'Hosur Road':1.0}
n=8000
lc=np.random.choice(locations,n)
bhk=np.random.choice([1,2,3,4,5],n,p=[0.1,0.35,0.35,0.15,0.05])
bath=np.clip(bhk+np.random.choice([0,1],n,p=[0.6,0.4]),1,6)
sqft=(bhk*np.random.uniform(450,700,n)).astype(int)
prices=[round((sqft[i]*4500*lm.get(lc[i],1.0)*np.random.uniform(0.85,1.15))/100000,2) for i in range(n)]
import os; os.makedirs('data',exist_ok=True)
pd.DataFrame({'area_type':np.random.choice(['Super built-up Area','Built-up Area','Plot Area','Carpet Area'],n),'availability':np.random.choice(['Ready To Move','6 months','1 Year'],n),'location':lc,'size':[f'{b} BHK' for b in bhk],'society':'','total_sqft':sqft,'bath':bath.astype(float),'balcony':np.random.choice([0,1,2,3],n).astype(float),'price':prices}).to_csv('data/bangalore_house_prices.csv',index=False)
print('Dataset created: data/bangalore_house_prices.csv')
# "