---
title: "Stuck on Android's 16KB Page Size Error? Here is the CMake Quick Fix"
published: false
description: "How to fix the '16KB page size support' issue required for Google Play Console submission when using Android NDK v27."
tags: "android, ndk, cmake, tutorial"
canonical_url: "https://zenn.dev/rg687076/articles/eb3e4f19a44c28"
---

# Abstract
- Support for 16KB page sizes is mandatory for publishing Android NDK apps on Google Play.
- How to support 16KB page sizes.
- How to verify 16KB page size alignment.

# Overview
Following the official Android developer documentation below, it can still be confusing to understand what steps to actually take:
https://developer.android.com/guide/practices/page-sizes#ndk-build

When I tried to publish an app for the first time in about 10 years, I found that the publication rules and procedures had changed significantly, which took a lot of effort. Among various requirements, like setting up a privacy policy, I got stuck on the **16KB page size support** issue.
This post shares how to resolve it.

## 16KB Page Size Support is Mandatory for Android NDK Apps
If your app does not support 16KB page sizes, uploading your AAB (or APK) to Google Play Console will fail.
Thus, you have no choice but to implement it.
The error looks like this:
![Error screen showing 4KB alignment mismatch](https://storage.googleapis.com/zenn-user-upload/01e88f8eb8bc-20251123.png)
*Notice the error: **4KB LOAD section alignment, but 16KB is required**.*

## How to Support 16KB Page Sizes
Normally, upgrading to NDK version 28 or higher is sufficient to resolve this issue. However, due to specific constraints, I had to stick to version 27.
In such cases, adding the following single line to your `CMakeLists.txt` resolves the problem:

```diff CMakeLists.txt
cmake_minimum_required(VERSION 3.22.1)

project("videophotobook")

add_library(VUFORIA_LIBRARY SHARED IMPORTED)
set_target_properties(VUFORIA_LIBRARY PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libVuforiaEngine.so)

add_library(${CMAKE_PROJECT_NAME} SHARED
        VuforiaController.cpp
        GLESRenderer.cpp
        GLESUtils.cpp
        Jni.cpp)

+ target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC include)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        log
        GLESv3
        VUFORIA_LIBRARY)
```

## How to Verify 16KB Page Size Support
Open Android Studio, select **Build** -> **Analyze APK...**, and inspect your built package.
![Analyze APK menu in Android Studio](https://storage.googleapis.com/zenn-user-upload/78a48223b71e-20251123.png)
You will see the details:
![Analyze APK results showing no 4KB alignment error](https://storage.googleapis.com/zenn-user-upload/20b25b0e2e96-20251123.png)
*If the error **4KB LOAD section alignment, but 16KB is required** is gone, it is OK.*

# Summary
Ideally, build your app using NDK version 28 or higher. If you must use an older NDK version, add `target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")` to your `CMakeLists.txt`.

I hope this helps!
