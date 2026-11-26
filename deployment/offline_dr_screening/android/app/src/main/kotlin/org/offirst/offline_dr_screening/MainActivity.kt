package org.offirst.offline_dr_screening

import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
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
                        } catch (_: Exception) {
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

    private fun report(call: MethodCall): File {
        val visit = requireNotNull(call.argument<String>("visit"))
        require(visit.matches(Regex("[A-Za-z0-9-]{1,120}")))
        val folder = File(filesDir, "screening_reports").apply { mkdirs() }
        val file = File(folder, "OFFIRST-$visit-v3.pdf")
        if (file.isFile) return file
        val heading = requireNotNull(call.argument<List<String>>("heading"))
        val images = requireNotNull(call.argument<List<Map<String, String>>>("images"))
        val staging = File.createTempFile("report-", ".staged", folder)
        try {
            val pdf = PdfDocument()
            try {
                fun page(number: Int, title: String, lines: List<String>, image: String? = null) {
                    val page = pdf.startPage(PdfDocument.PageInfo.Builder(595, 842, number).create())
                    val canvas = page.canvas
                    val ink = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.rgb(0, 51, 153)
                        typeface = Typeface.DEFAULT_BOLD
                        textSize = 24f
                    }
                    val header = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.rgb(0, 51, 153)
                    }
                    canvas.drawRect(0f, 0f, 595f, 92f, header)
                    val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.WHITE
                        typeface = Typeface.DEFAULT_BOLD
                        textSize = 23f
                    }
                    canvas.drawText("OFFIRST", 42f, 39f, titlePaint)
                    titlePaint.textSize = 15f
                    titlePaint.typeface = Typeface.DEFAULT
                    canvas.drawText(title, 42f, 68f, titlePaint)
                    val content = lines.joinToString("\n\n")
                    val text = StaticLayout.Builder.obtain(
                        content, 0, content.length,
                        TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
                            color = Color.rgb(26, 34, 48)
                            textSize = 14f
                        }, 511
                    ).setAlignment(Layout.Alignment.ALIGN_NORMAL).build()
                    canvas.save()
                    canvas.translate(42f, 112f)
                    text.draw(canvas)
                    canvas.restore()
                    if (image != null) {
                        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                        BitmapFactory.decodeFile(image, bounds)
                        var sample = 1
                        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 1024) sample *= 2
                        val bitmap = BitmapFactory.decodeFile(image,
                            BitmapFactory.Options().apply { inSampleSize = sample })
                        val top = 124f + text.height
                        if (bitmap != null) {
                            val scale = minOf(511f / bitmap.width, (790f - top) / bitmap.height)
                            if (scale > 0) {
                                canvas.drawBitmap(bitmap, null, RectF(42f, top,
                                    42f + bitmap.width * scale, top + bitmap.height * scale),
                                    Paint(Paint.FILTER_BITMAP_FLAG))
                            }
                        } else {
                            canvas.drawText("Original image unavailable", 42f, top + 20f, ink)
                        }
                    }
                    ink.textSize = 10f
                    canvas.drawText("OFFIRST  |  Experimental AI screening  |  $number", 42f, 811f, ink)
                    pdf.finishPage(page)
                }
                page(1, "OFFIRST screening report", heading)
                images.forEachIndexed { index, row ->
                    page(index + 2, row.getValue("eye"), listOf(
                        "Image: ${row.getValue("id")}",
                        "Recorded: ${row.getValue("date")}",
                        "AI output: ${row.getValue("result")}",
                        "Classification: ${row.getValue("classification")}",
                        "Image quality: ${row.getValue("quality")}",
                        "Model: ${row.getValue("model")}"
                    ), row["path"])
                }
                staging.outputStream().use { pdf.writeTo(it) }
            } finally {
                pdf.close()
            }
            check(staging.renameTo(file))
            return file
        } finally {
            staging.delete()
        }
    }

    companion object {
        private const val EXPORT_PDF = 101
    }
}
