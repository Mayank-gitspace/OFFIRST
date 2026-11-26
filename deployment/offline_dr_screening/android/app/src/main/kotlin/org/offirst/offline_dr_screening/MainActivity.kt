package org.offirst.offline_dr_screening

import android.app.Activity
import android.content.Intent
import androidx.core.content.FileProvider
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File

class MainActivity : FlutterActivity() {
    private var pendingExport: Pair<File, MethodChannel.Result>? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "offirst/v2_1_preprocessing")
            .setMethodCallHandler { call, result ->
                if (call.method != "prepareTensor") {
                    result.notImplemented()
                } else {
                    val encoded = call.arguments as? ByteArray
                    if (encoded == null || encoded.size > 32 * 1024 * 1024) {
                        result.error("image", "Invalid or oversized image.", null)
                    } else {
                        Thread {
                            try {
                                val tensor = V21Preprocessor.prepare(encoded)
                                runOnUiThread { result.success(tensor) }
                            } catch (error: Exception) {
                                val stage = (error as? ImagePipelineFailure)?.stage ?: "preprocess"
                                logPipelineFailure(stage, error)
                                val message = (error as? ImagePipelineFailure)?.message ?: "Image preprocessing failed."
                                runOnUiThread { result.error(stage, message, null) }
                            } catch (error: LinkageError) {
                                logPipelineFailure("runtime", error)
                                runOnUiThread { result.error("runtime", "OpenCV native runtime could not load.", null) }
                            } catch (error: OutOfMemoryError) {
                                logPipelineFailure("image_size", error)
                                runOnUiThread { result.error("image_size", "Insufficient memory for this image. Select a smaller image.", null) }
                            }
                        }.start()
                    }
                }
            }
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "offirst/reports")
            .setMethodCallHandler { call, result ->
                if (call.method !in setOf("save", "view", "export")) {
                    result.notImplemented()
                } else if (pendingExport != null) {
                    result.error("busy", "A PDF export is already open.", null)
                } else {
                    Thread {
                        try {
                            val file = report(call)
                            runOnUiThread { openReport(call.method, file, result) }
                        } catch (error: Exception) {
                            val frames = error.stackTrace.take(8).joinToString(" > ") {
                                "${it.methodName}:${it.lineNumber}"
                            }
                            android.util.Log.e(
                                "OFFIRST_REPORT",
                                "type=${error.javaClass.simpleName} $frames",
                            )
                            runOnUiThread { result.error("report", "Could not create PDF.", null) }
                        }
                    }.start()
                }
            }
    }

    private fun logPipelineFailure(stage: String, error: Throwable) {
        // Never log exception messages, image bytes, paths, or patient metadata.
        val frames = error.stackTrace.take(6).joinToString(" > ") {
            "${it.className}.${it.methodName}:${it.lineNumber}"
        }
        android.util.Log.e("OFFIRST_INFERENCE", "stage=$stage type=${error.javaClass.simpleName} $frames")
    }

    private fun openReport(action: String, file: File, result: MethodChannel.Result) {
        try {
            when (action) {
                "view" -> {
                    val uri = FileProvider.getUriForFile(this, "$packageName.reports", file)
                    startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(uri, "application/pdf")
                        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
                    result.success(true)
                }
                "export" -> {
                    val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = "application/pdf"
                        putExtra(Intent.EXTRA_TITLE, file.name)
                    }
                    startActivityForResult(intent, EXPORT_PDF)
                    pendingExport = file to result
                }
                else -> result.success(true)
            }
        } catch (_: Exception) {
            result.error("open", "Could not open the PDF action.", null)
        }
    }

    @Deprecated("Receives the result from the Android document picker.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != EXPORT_PDF) return
        val (file, result) = pendingExport ?: return
        pendingExport = null
        val uri = data?.data
        if (resultCode != Activity.RESULT_OK || uri == null) {
            result.success(false)
            return
        }
        Thread {
            try {
                val output = requireNotNull(contentResolver.openOutputStream(uri))
                output.use { stream -> file.inputStream().use { it.copyTo(stream) } }
                runOnUiThread { result.success(true) }
            } catch (_: Exception) {
                runOnUiThread { result.error("export", "Could not export PDF.", null) }
            }
        }.start()
    }

    private fun report(call: MethodCall): File = ReportPdf.write(this, call)

    companion object {
        private const val EXPORT_PDF = 101
    }
}