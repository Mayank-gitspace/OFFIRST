package org.offirst.offline_dr_screening

import android.graphics.BitmapFactory
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

class ImagePipelineFailure(val stage: String, message: String) : Exception(message)

/** Frozen v2.1 OpenCV 4.12 preprocessing; no calibration or model decision here. */
object V21Preprocessor {
    fun prepare(encoded: ByteArray): ByteArray {
        if (encoded.isEmpty() || encoded.size > 32 * 1024 * 1024) {
            throw ImagePipelineFailure("image_access", "Image is empty or exceeds 32 MiB.")
        }
        val jpeg = encoded.size >= 2 && encoded[0] == 0xff.toByte() && encoded[1] == 0xd8.toByte()
        val png = encoded.size >= 8 && encoded.take(8).toByteArray().contentEquals(
            byteArrayOf(0x89.toByte(), 0x50, 0x4e, 0x47, 13, 10, 26, 10))
        if (!jpeg && !png) throw ImagePipelineFailure("image_format", "Select a JPEG or PNG image.")
        // Bounds-only inspection prevents a compressed image allocating an unbounded Mat.
        // Accepted images are decoded at their original resolution, never downsampled here.
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(encoded, 0, encoded.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw ImagePipelineFailure("image_decode", "Image decoding failed.")
        }
        if (bounds.outWidth.toLong() * bounds.outHeight > 24_000_000L) {
            throw ImagePipelineFailure("image_size", "Image exceeds the 24 megapixel memory limit.")
        }
        if (!OpenCVLoader.initLocal()) throw ImagePipelineFailure("runtime", "OpenCV could not initialize.")
        val owned = ArrayList<Mat>()
        fun <T : Mat> own(mat: T): T { owned.add(mat); return mat }
        try {
            val source = own(Mat(1, encoded.size, CvType.CV_8UC1))
            check(source.put(0, 0, encoded) == encoded.size)
            // IMREAD_COLOR applies OpenCV EXIF orientation just like frozen Python imread.
            val bgr = own(Imgcodecs.imdecode(source, Imgcodecs.IMREAD_COLOR))
            if (bgr.empty()) throw ImagePipelineFailure("image_decode", "OpenCV image decoding failed.")
            val rgb = own(Mat())
            Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB)
            bgr.release(); source.release()
            val h = rgb.rows(); val w = rgb.cols()
            val scale = 512.0 / maxOf(h, w)
            val proxy = own(Mat())
            Imgproc.resize(rgb, proxy, Size(maxOf(1, (w * scale).toInt()).toDouble(),
                maxOf(1, (h * scale).toInt()).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            val channels = ArrayList<Mat>(3)
            Core.split(proxy, channels)
            channels.forEach { own(it) }
            val maxChannel = own(Mat())
            Core.max(channels[0], channels[1], maxChannel)
            Core.max(maxChannel, channels[2], maxChannel)
            val mask = own(Mat())
            Imgproc.threshold(maxChannel, mask, 15.0, 1.0, Imgproc.THRESH_BINARY)
            val kernel = own(Mat.ones(5, 5, CvType.CV_8U))
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)
            requireFundusField(maxChannel)
            if (Core.countNonZero(mask) >= 100) {
                val points = own(MatOfPoint())
                Core.findNonZero(mask, points)
                val box = Imgproc.boundingRect(points)
                val x0 = (box.x / scale).toInt().coerceIn(0, w)
                val x1 = ((box.x + box.width) / scale).toInt().coerceIn(0, w)
                val y0 = (box.y / scale).toInt().coerceIn(0, h)
                val y1 = ((box.y + box.height) / scale).toInt().coerceIn(0, h)
                if (x1 > x0 && y1 > y0) {
                    val roi = own(Mat(rgb, Rect(x0, y0, x1 - x0, y1 - y0)))
                    val crop = own(roi.clone())
                    roi.release(); rgb.release()
                    val side = maxOf(crop.rows(), crop.cols())
                    val top = (side - crop.rows()) / 2
                    val left = (side - crop.cols()) / 2
                    Core.copyMakeBorder(crop, rgb, top, side - crop.rows() - top,
                        left, side - crop.cols() - left, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))
                    crop.release()
                }
            }
            val cache = own(Mat()); val work = own(Mat())
            Imgproc.resize(rgb, cache, Size(640.0, 640.0), 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.resize(cache, work, Size(512.0, 512.0), 0.0, 0.0, Imgproc.INTER_AREA)
            rgb.release(); cache.release()
            val workFloat = own(Mat())
            work.convertTo(workFloat, CvType.CV_32FC3)
            val interleaved = FloatArray(512 * 512 * 3)
            // OpenCV Java Mat.get(float[]) returns BYTES, not the number of floats.
            check(workFloat.get(0, 0, interleaved) == interleaved.size * 4) {
                "Incomplete float matrix read."
            }
            val output = ByteBuffer.allocate(interleaved.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            val means = floatArrayOf(0.485f, 0.456f, 0.406f)
            val stds = floatArrayOf(0.229f, 0.224f, 0.225f)
            for (channel in 0..2) for (index in 0 until 512 * 512) {
                output.putFloat((interleaved[index * 3 + channel] - means[channel] * 255f) /
                    (stds[channel] * 255f))
            }
            return output.array()
        } finally {
            // release() is idempotent; this also covers partially completed preprocessing.
            owned.asReversed().forEach { it.release() }
        }
    }

    /**
     * Rejects a photo that has no retinal disc. The surround is measured from
     * the corners, so a gray camera border still passes. One labeled corner is
     * allowed. This does not change the crop mask, so an accepted image keeps
     * the frozen tensor.
     */
    private fun requireFundusField(maxChannel: Mat) {
        val owned = ArrayList<Mat>()
        fun <T : Mat> own(mat: T): T {
            owned.add(mat)
            return mat
        }
        try {
            val height = maxChannel.rows()
            val width = maxChannel.cols()
            val pixels = height.toLong() * width
            if (pixels <= 0L) throw notFundus()
            val shorter = minOf(height, width)
            val patch = maxOf(1, minOf(shorter / 2, maxOf(1, shorter / 28)))
            val cornerMeans = ArrayList<Double>(4)
            for ((x, y) in listOf(
                0 to 0,
                width - patch to 0,
                0 to height - patch,
                width - patch to height - patch,
            )) {
                val roi = own(Mat(maxChannel, Rect(x, y, patch, patch)))
                cornerMeans.add(Core.mean(roi).`val`[0])
            }
            val sorted = cornerMeans.sorted()
            val surround = (sorted[1] + sorted[2]) / 2.0
            val darkCorners = cornerMeans.count { it <= 110.0 }
            val gate = own(Mat())
            Imgproc.threshold(
                maxChannel, gate, maxOf(18.0, surround + 8.0), 1.0, Imgproc.THRESH_BINARY)
            val kernel = own(Mat.ones(5, 5, CvType.CV_8U))
            Imgproc.morphologyEx(gate, gate, Imgproc.MORPH_OPEN, kernel)
            val foreground = Core.countNonZero(gate)
            val fraction = foreground.toDouble() / pixels
            val interior = if (foreground > 0) Core.mean(maxChannel, gate).`val`[0] else 0.0
            val moments = Imgproc.moments(gate, true)
            val centered = moments.m00 > 0.0 &&
                abs(moments.m10 / moments.m00 - width / 2.0) / width <= 0.34 &&
                abs(moments.m01 / moments.m00 - height / 2.0) / height <= 0.34
            val labels = own(Mat())
            val stats = own(Mat())
            val centroids = own(Mat())
            val components = Imgproc.connectedComponentsWithStats(
                gate, labels, stats, centroids, 8, CvType.CV_32S)
            var largest = 0
            for (label in 1 until components) {
                largest = maxOf(largest, stats.get(label, Imgproc.CC_STAT_AREA)[0].toInt())
            }
            val dominant = foreground > 0 && largest.toDouble() / foreground >= 0.55
            if (darkCorners < 3 || fraction !in 0.07..0.97 || interior < surround + 12.0 ||
                !centered || !dominant) {
                throw notFundus()
            }
        } finally {
            owned.asReversed().forEach { it.release() }
        }
    }

    private fun notFundus() = ImagePipelineFailure(
        "not_fundus",
        "This image does not look like a fundus photograph. Choose a retinal image.",
    )
}
