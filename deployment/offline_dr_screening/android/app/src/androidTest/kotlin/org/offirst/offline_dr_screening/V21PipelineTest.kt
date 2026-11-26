package org.offirst.offline_dr_screening

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.GZIPInputStream
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class V21PipelineTest {
    @Test fun floatMatReadReturnsBytes() {
        assertTrue(OpenCVLoader.initLocal())
        val mat = Mat.ones(1, 1, CvType.CV_32FC3)
        try {
            val values = FloatArray(3)
            val copied = mat.get(0, 0, values)
            assertEquals(12, copied)
            assertNotEquals(values.size, copied)
        } finally { mat.release() }
    }

    @Test fun unsupportedAndCorruptImagesReportTheirStage() {
        for ((bytes, stage) in listOf(byteArrayOf() to "image_access",
            byteArrayOf(1,2,3) to "image_format",
            byteArrayOf(0xff.toByte(),0xd8.toByte(),0xff.toByte(),0xd9.toByte()) to "image_decode")) {
            try { V21Preprocessor.prepare(bytes); fail("Expected image rejection") }
            catch (error: ImagePipelineFailure) { assertEquals(stage, error.stage) }
        }
    }

    @Test fun retinalDiscPassesAndOrdinaryPhotosAreRejected() {
        assertTrue(OpenCVLoader.initLocal())
        for (encoded in listOf(
            jpeg(1024, 768) { image ->
                Imgproc.ellipse(image, Point(512.0, 384.0), Size(384.0, 307.2),
                    0.0, 0.0, 360.0, Scalar(35.0, 65.0, 145.0), -1)
            },
            jpeg(1024, 768) { image ->
                image.setTo(Scalar(70.0, 70.0, 70.0))
                Imgproc.circle(image, Point(512.0, 384.0), 320, Scalar(40.0, 80.0, 160.0), -1)
            },
            jpeg(1000, 1000) { image ->
                Imgproc.circle(image, Point(500.0, 500.0), 490, Scalar(30.0, 60.0, 140.0), -1)
            },
            jpeg(1024, 768) { image ->
                Imgproc.circle(image, Point(512.0, 384.0), 340, Scalar(40.0, 80.0, 160.0), -1)
                Imgproc.rectangle(image, Point(0.0, 0.0), Point(70.0, 36.0), Scalar(240.0, 240.0, 240.0), -1)
            },
        )) {
            assertEquals(3 * 512 * 512 * 4, V21Preprocessor.prepare(encoded).size)
        }
        for (encoded in listOf(
            jpeg(640, 480) { it.setTo(Scalar(60.0, 100.0, 180.0)) },
            jpeg(800, 600) { it.setTo(Scalar(245.0, 245.0, 245.0)) },
            jpeg(1024, 768) {
                Imgproc.circle(it, Point(512.0, 384.0), 12, Scalar(40.0, 80.0, 200.0), -1)
            },
            jpeg(1024, 768) {
                Imgproc.circle(it, Point(180.0, 140.0), 120, Scalar(40.0, 70.0, 160.0), -1)
            },
        )) {
            try {
                V21Preprocessor.prepare(encoded)
                fail("Expected a non-fundus image to be rejected")
            } catch (error: ImagePipelineFailure) {
                assertEquals("not_fundus", error.stage)
            }
        }
    }

    private fun jpeg(width: Int, height: Int, draw: (Mat) -> Unit): ByteArray {
        val image = Mat(height, width, CvType.CV_8UC3, Scalar(0.0, 0.0, 0.0))
        val encoded = MatOfByte()
        try {
            draw(image)
            assertTrue(Imgcodecs.imencode(".jpg", image, encoded))
            return encoded.toArray()
        } finally {
            image.release()
            encoded.release()
        }
    }

    @Test fun allSyntheticAndroidTensorsMatchFrozenPythonReference() {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val manifest = JSONObject(assets.open("manifest.json").bufferedReader().use { it.readText() })
        val fixtures = manifest.getJSONArray("fixtures")
        assertEquals(5, fixtures.length())
        for (i in 0 until fixtures.length()) {
            val row = fixtures.getJSONObject(i)
            val name = row.getString("name")
            val actualBytes = V21Preprocessor.prepare(assets.open(name).use { it.readBytes() })
            val expectedBytes = GZIPInputStream(assets.open("$name.f32.gz")).use { it.readBytes() }
            assertEquals(expectedBytes.size, actualBytes.size)
            val actual = ByteBuffer.wrap(actualBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val expected = ByteBuffer.wrap(expectedBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            var maxError = 0f
            for (index in 0 until actual.limit()) maxError = maxOf(maxError, abs(actual.get(index)-expected.get(index)))
            assertTrue("$name max tensor error=$maxError", maxError <= 1e-5f)
        }
    }
}
