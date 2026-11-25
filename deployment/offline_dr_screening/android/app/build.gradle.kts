import java.util.Base64

plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

val offirstV21Build = providers.gradleProperty("dart-defines").orNull
    ?.split(",")
    ?.any { encoded ->
        runCatching {
            String(Base64.getDecoder().decode(encoded), Charsets.UTF_8)
        }.getOrNull() == "OFFIRST_EXPERIMENTAL_V21=true"
    } == true

android {
    namespace = "org.offirst.offline_dr_screening"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    if (!offirstV21Build) {
        externalNativeBuild {
            cmake {
                path = file("../../native/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    packaging {
        jniLibs {
            // Compress the sideload APK. Android extracts the native libraries at install time.
            useLegacyPackaging = true
            // Flutter native assets can include JNI helpers outside the supported ARM ABIs.
            excludes += setOf("lib/x86/**", "lib/x86_64/**")
        }
    }

    defaultConfig {
        applicationId = "org.offirst.offline_dr_screening"
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        if (!offirstV21Build) {
            externalNativeBuild {
                cmake { targets += "offirst_jpeg" }
            }
        }
        ndk {
            // Flutter supplies the filters for split APKs. AGP rejects both mechanisms together.
            if (providers.gradleProperty("split-per-abi").orNull != "true") {
                abiFilters += listOf("arm64-v8a", "armeabi-v7a")
            }
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // OpenCV 4.12.0 mirrors the Python development pipeline used by v2.1.
    // It is packaged in the APK; inference and preprocessing remain offline.
    implementation("org.opencv:opencv:4.12.0")
}
