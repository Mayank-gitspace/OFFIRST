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
