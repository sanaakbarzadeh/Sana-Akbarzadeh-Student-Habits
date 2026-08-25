#ثنا اکبرزاده پروژه یک ( پیش بینی نمره امتحان بر اساس عادات روزمره)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

import joblib



# مرحله ۱: بارگذاری و بررسی داده


df = pd.read_csv("enhanced_student_habits_performance_dataset.csv")

print("اطلاعات دیتاست:")
print(df.info())

print("\nآمار توصیفی:")
print(df.describe())

print("\nتعداد ردیف‌ها:", df.shape[0])
print("تعداد ستون‌ها:", df.shape[1])



# مرحله ۲: پاک‌سازی داده

print("\nداده‌های گمشده در هر ستون:")
print(df.isnull().sum())

print("\nتعداد ردیف‌های تکراری:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nتعداد ردیف‌ها بعد از حذف داده‌های تکراری:", df.shape[0])



# مرحله ۳: تبدیل داده‌های غیرعددی به عددی


binary_columns = [
    "part_time_job",
    "extracurricular_participation",
    "access_to_tutoring",
    "dropout_risk"
]

for column in binary_columns:
    df[column] = df[column].map({
        "Yes": 1,
        "No": 0
    })


categorical_columns = [
    "gender",
    "major",
    "learning_style",
    "diet_quality",
    "parental_education_level",
    "internet_quality",
    "study_environment",
    "family_income_range"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)

print("\nاطلاعات دیتاست بعد از تبدیل داده‌های غیرعددی:")
print(df.info())

print("\nتعداد ردیف‌ها و ستون‌ها بعد از تبدیل:")
print("تعداد ردیف‌ها:", df.shape[0])
print("تعداد ستون‌ها:", df.shape[1])



# مرحله ۴: رسم نمودارها


# نمودار ۱: توزیع نمره امتحان

plt.figure(figsize=(8, 5))

plt.hist(
    df["exam_score"],
    bins=20
)

plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Exam Scores")

plt.show()


# نمودار ۲: رابطه ساعات مطالعه و نمره امتحان

plt.figure(figsize=(8, 5))

plt.scatter(
    df["study_hours_per_day"],
    df["exam_score"],
    alpha=0.3
)

plt.xlabel("Study Hours per Day")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")

plt.show()


# نمودار ۳: مقایسه نمره امتحان بر اساس جنسیت

gender_data = pd.read_csv(
    "enhanced_student_habits_performance_dataset.csv"
)

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="gender",
    y="exam_score",
    data=gender_data
)

plt.xlabel("Gender")
plt.ylabel("Exam Score")
plt.title("Exam Score by Gender")

plt.show()



# مرحله ۵: تقسیم داده به آموزش و تست


X = df.drop(
    ["exam_score", "student_id"],
    axis=1
)

y = df["exam_score"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nمرحله ۵: تقسیم داده‌ها")

print(
    "تعداد داده‌های آموزشی:",
    len(X_train)
)

print(
    "تعداد داده‌های تست:",
    len(X_test)
)



# مرحله ۶: ساخت و آموزش مدل


model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nمرحله ۶: ساخت مدل")

print(
    "مدل Decision Tree با موفقیت آموزش داده شد."
)



# مرحله ۷: ارزیابی مدل


y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

print("\nمرحله ۷: ارزیابی مدل")

print(
    "MAE مدل:",
    mae
)

if mae <= 4.0:

    print(
        "مدل به حد مجاز خطای 4.0 رسیده است."
    )

else:

    print(
        "خطای مدل بیشتر از 4.0 است و نیاز به تنظیم مدل دارد."
    )


# مرحله ۸: نمودار اهمیت ویژگی‌ها


feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nمرحله ۸: اهمیت ویژگی‌ها")

print("\n3 ویژگی مهم:")
print(feature_importance.head(3))

# نمودار اهمیت 10 ویژگی برتر
plt.figure(figsize=(10, 6))
feature_importance.head(10).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importance")
plt.tight_layout()
plt.show()


# مرحله ۹: ذخیره مدل


joblib.dump(
    model,
    "student_habits_model.joblib"
)

print("\nمرحله ۹: ذخیره‌سازی مدل")

print(
    "مدل با موفقیت در فایل student_habits_model.joblib ذخیره شد."
)
