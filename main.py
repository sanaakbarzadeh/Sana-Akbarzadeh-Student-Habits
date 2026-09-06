#ثنا اکبرزاده (پروژه یک_ پیش بینی نمره دانش آموزان بر اساس عادات روزمره)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

import joblib



# عنوان پروژه


st.title("پروژه شماره ۱ - پیش‌بینی نمره امتحان بر اساس عادات روزمره")



# مرحله ۱: بارگذاری و بررسی داده


st.header("مرحله ۱: بارگذاری و بررسی داده")

df = pd.read_csv("enhanced_student_habits_performance_dataset.csv")

st.subheader("اطلاعات دیتاست")

st.subheader("اطلاعات دیتاست")

info_df = pd.DataFrame({
    "نوع داده": df.dtypes.astype(str),
    "تعداد مقادیر غیرخالی": df.notna().sum(),
    "تعداد مقادیر خالی": df.isna().sum()
})

st.dataframe(info_df, use_container_width=True)

st.subheader("آمار توصیفی")
st.dataframe(df.describe())

st.write("تعداد ردیف‌ها:", df.shape[0])
st.write("تعداد ستون‌ها:", df.shape[1])



# مرحله ۲: بررسی داده‌های گمشده و تکراری


st.header("مرحله ۲: پاک‌سازی داده‌ها")

st.subheader("داده‌های گمشده در هر ستون")
st.dataframe(df.isnull().sum().to_frame("تعداد داده‌های گمشده"))

st.subheader("تعداد ردیف‌های تکراری")
st.write(df.duplicated().sum())

df = df.drop_duplicates()

st.write(
    "تعداد ردیف‌ها بعد از حذف داده‌های تکراری:",
    df.shape[0]
)



# مرحله ۳: تبدیل داده‌های غیرعددی


st.header("مرحله ۳: تبدیل داده‌های غیرعددی")

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

st.subheader("اطلاعات دیتاست بعد از تبدیل داده‌های غیرعددی")

info_df_after = pd.DataFrame({
    "نوع داده": df.dtypes.astype(str),
    "تعداد مقادیر غیرخالی": df.notna().sum(),
    "تعداد مقادیر خالی": df.isna().sum()
})

st.dataframe(info_df_after, use_container_width=True)

st.write("تعداد ردیف‌ها:", df.shape[0])
st.write("تعداد ستون‌ها:", df.shape[1])



# مرحله ۴: نمودارها


st.header("مرحله ۴: نمودارهای بررسی داده")


# نمودار ۱: توزیع نمره امتحان

st.subheader("توزیع نمره‌های امتحان")

fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.hist(
    df["exam_score"],
    bins=20
)

ax1.set_xlabel("Exam Score")
ax1.set_ylabel("Number of Students")
ax1.set_title("Distribution of Exam Scores")

st.pyplot(fig1)


# نمودار ۲: ساعت مطالعه و نمره

st.subheader("رابطه ساعت مطالعه و نمره امتحان")

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.scatter(
    df["study_hours_per_day"],
    df["exam_score"],
    alpha=0.3
)

ax2.set_xlabel("Study Hours per Day")
ax2.set_ylabel("Exam Score")
ax2.set_title("Study Hours vs Exam Score")

st.pyplot(fig2)


# نمودار ۳: نمره بر اساس جنسیت

st.subheader("نمره امتحان بر اساس جنسیت")

gender_data = pd.read_csv(
    "enhanced_student_habits_performance_dataset.csv"
)

fig3, ax3 = plt.subplots(figsize=(8, 5))

sns.boxplot(
    x="gender",
    y="exam_score",
    data=gender_data,
    ax=ax3
)

ax3.set_xlabel("Gender")
ax3.set_ylabel("Exam Score")
ax3.set_title("Exam Score by Gender")

st.pyplot(fig3)



# مرحله ۵: تقسیم داده‌ها


st.header("مرحله ۵: تقسیم داده‌ها به آموزشی و تست")

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

st.write(
    "تعداد داده‌های آموزشی:",
    len(X_train)
)

st.write(
    "تعداد داده‌های تست:",
    len(X_test)
)



# مرحله ۶: ساخت مدل


st.header("مرحله ۶: ساخت مدل Decision Tree")

model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

st.success(
    "مدل DecisionTree با موفقیت آموزش داده شد."
)



# مرحله ۷: ارزیابی مدل


st.header("مرحله ۷: ارزیابی مدل")

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

st.write("MAE مدل:", mae)

if mae <= 4.0:
    st.success(
        "مدل به حد مجاز خطای 4.0 رسیده است."
    )
else:
    st.error(
        "خطای مدل بیشتر از 4.0 است و نیاز به تنظیم مدل دارد."
    )



# مرحله ۸: اهمیت ویژگی‌ها

st.header("مرحله ۸: اهمیت ویژگی‌ها")

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)


st.subheader("3 ویژگی مهم")

st.dataframe(
    feature_importance.head(3).to_frame("Importance")
)


# نمودار Feature Importance

st.subheader("Top 10 Feature Importance")

top_10 = feature_importance.head(10).sort_values()

fig4, ax4 = plt.subplots(figsize=(10, 6))

ax4.barh(
    top_10.index,
    top_10.values
)

ax4.set_xscale("symlog", linthresh=0.0001)

ax4.set_xlabel("Importance")
ax4.set_ylabel("Feature")
ax4.set_title("Top 10 Feature Importance")

ax4.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

st.pyplot(fig4)



# مرحله ۹: ذخیره مدل


st.header("مرحله ۹: ذخیره‌سازی مدل")

joblib.dump(
    model,
    "student_habits_model.joblib"
)

st.success(
    "مدل با موفقیت در فایل student_habits_model.joblib ذخیره شد."
)
