# Keep the method-channel classes and the native ML entry points R8 must not strip.
-keep class org.offirst.offline_dr_screening.** { *; }

-keep class org.opencv.android.OpenCVLoader { *; }
-keepclasseswithmembernames,includedescriptorclasses class org.opencv.** {
    native <methods>;
}

-keep class com.github.dart_lang.jni.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
